"""Evidence-based GPU selection: never use long jobs as exploratory pilots.

Known EB01 2026-11-13 incidents:
- Wan2.2 TI2V-5B at 320x576 / 20-24 steps produced severe RGB artifacts;
  technical ffmpeg success was NOT aesthetic quality.
- LTX-Video 2B (legacy) initial full 125f session OOM on Kaggle T4 14.56 GiB;
  staged offload avoided initial OOM but full 125f still timed out at 2600s.

This is a PRE-FLIGHT resource/policy gate, not a promise of visual quality.
Never automatically initiate paid generation or claim a model is supported
because its weights are downloadable.
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Any

_SHA = re.compile(r"^[a-f0-9]{64}$")
MODEL_WAN = "wan22-ti2v-5b"
MODEL_LTX_LEGACY = "ltx-video-2b-distilled"
MODEL_LTX_NEW = "ltx-2.5-22b"


@dataclass(frozen=True)
class RenderRequest:
    model: str
    gpu_vram_gib: float
    width: int
    height: int
    frames: int
    steps: int
    mode: str = "production"
    max_gpu_seconds: int = 2700
    paid_service: bool = False
    license_confirmed: bool = False


def check_candidate(request: RenderRequest, pilot: dict[str, Any] | None = None) -> dict:
    """Return audited verdict. All automatic decisions FAIL CLOSED."""
    reasons = []
    vram = (request.gpu_vram_gib if type(request.gpu_vram_gib) in (int,float)
            and math.isfinite(request.gpu_vram_gib) else 0)
    width = request.width if type(request.width) is int else 0
    height = request.height if type(request.height) is int else 0
    frames = request.frames if type(request.frames) is int else 0

    def block(msg: str) -> None:
        if msg not in reasons:
            reasons.append(msg)

    if request.model not in (MODEL_WAN, MODEL_LTX_LEGACY, MODEL_LTX_NEW):
        block("UNSUPPORTED_UNREVIEWED_MODEL")
    if request.mode not in ("pilot", "production"):
        block("UNKNOWN_RUN_MODE")
    if not isinstance(request.gpu_vram_gib, (int,float)) or (
        not math.isfinite(request.gpu_vram_gib) or request.gpu_vram_gib < 1):
        block("INVALID_GPU_CAPACITY")
    if any(type(x) is not int for x in
           (request.width, request.height, request.frames, request.steps,request.max_gpu_seconds)):
        block("INVALID_JOB_GEOMETRY_OR_TIMEOUT")
    elif (request.width < 256 or request.height < 448 or
          request.frames < 9 or request.steps < 1 or
          request.max_gpu_seconds < 120 or request.max_gpu_seconds > 3600):
        block("INVALID_JOB_GEOMETRY_OR_TIMEOUT")

    # Distilled LTX 2B config in pinned official source runs exactly 8
    # inference steps. Before this fix the preflight falsely claimed 20
    # steps while the actual model ran 8, invalidating resource estimates.
    if request.model == MODEL_LTX_LEGACY:
        if type(request.steps) is not int or request.steps != 8:
            block("LTX_2B_DISTILLED_REQUIRES_ACTUAL_8_STEPS")
    elif type(request.steps) is not int or request.steps < 10:
        block("NON_DISTILLED_MODEL_REQUIRES_MINIMUM_10_STEPS")

    if request.paid_service is not False:
        block("PAID_SERVICE_NOT_AUTHORIZED")
    if request.license_confirmed is not True:
        block("MODEL_LICENCE_NOT_CONFIRMED")
    if request.model == MODEL_WAN:
        if width % 32 or height % 32 or (frames-1)%4:
            block("WAN_5B_REQUIRES_32PX_SPATIAL_AND_4N_PLUS_1_TEMPORAL_ALIGNMENT")
        if request.mode == "production":
            if vram < 24:
                block("WAN_720P_OFFLOAD_REQUIRES_AT_LEAST_24GIB_REVIEWED_GPU")
            if width < 704 or height < 1280:
                block("WAN_PREVIOUS_LOW_RES_320X576_WAS_ARTISTICALLY_REJECTED")
        else:
            if frames > 49:
                block("PROOF_OF_QUALITY_FIRST_LIMIT_PILOT_TO_49_NATIVE_FRAMES")
    elif request.model == MODEL_LTX_LEGACY:
        if request.mode == "production":
            if vram < 24:
                block("LTX_125F_T4_2600S_TIMEOUT_KNOWN_NEEDS_NEW_HARDWARE_OR_BENCHMARK")
            if frames > 49 and vram < 24:
                block("REPEATING_KNOWN_TIMED_OUT_125F_T4_JOB_FORBIDDEN")
        else:
            if frames > 33:
                block("LEGACY_LTX_UNBENCHMARKED_GPU_PILOT_MAX_33_FRAMES")
    elif request.model == MODEL_LTX_NEW:
        if vram < 24:
            block("LTX25_PRODUCTION_HARDWARE_BELOW_PUBLISHED_DESKTOP_FLOOR")
        if request.mode == "pilot" and frames > 49:
            block("LTX25_PILOT_MUST_BE_SHORT")

    if request.mode == "production":
        if not isinstance(pilot, dict):
            block("NO_PREVIOUS_VERIFIED_TRUE_GPU_SHORT_PILOT")
        else:
            if pilot.get("model") != request.model:
                block("PILOT_MODEL_DIFFERS_FROM_PRODUCTION")
            if pilot.get("source_neural_inference_performed") is not True or not (
                isinstance(pilot.get("clip_sha256"),str) and _SHA.fullmatch(pilot["clip_sha256"])):
                block("PILOT_ORIGINAL_NATIVE_GPU_SOURCE_NOT_SHA_VERIFIED")
            if pilot.get("human_artistic_review") != "PASS":
                block("PILOT_NOT_HUMAN_APPROVED_FOR_IDENTITY_AND_MOTION")
            if pilot.get("rgb_artifact_gate") != "PASS":
                block("PILOT_RGB_COLOR_QA_NOT_PASS")
            # Never guess per-frame speed from an OOM/timeout failure.
            sec,frames = pilot.get("actual_gpu_seconds"),pilot.get("actual_neural_frames")
            if (type(sec) not in (int,float) or not math.isfinite(sec) or sec <= 0
                    or type(frames) is not int or not 9 <= frames <= 49):
                block("PILOT_ACTUAL_GPU_SPEED_UNMEASURED")
            elif (type(request.frames) is int and type(request.max_gpu_seconds) is int
                  and sec * request.frames / frames * 1.75 > request.max_gpu_seconds):
                block("ESTIMATED_JOB_EXCEEDS_GPU_TIME_BUDGET_NO_LONG_RETRY")
    return {
        "schema":1,
        "status":"BLOCKED" if reasons else (
            "PILOT_ALLOWED_NOT_QUALITY_APPROVED" if request.mode=="pilot"
            else "PRECHECK_PASSED_FINAL_MOVIE_STILL_NEEDS_ARTISTIC_QA"),
        "reasons":reasons,
        "model":request.model,
        "current_gpu_vram_gib":request.gpu_vram_gib,
        "frames_requested":request.frames,
        "production_approved":not reasons and request.mode=="production",
        "quality_is_proven":False,
        "gpu_launched":False,
        "published":False,
    }
