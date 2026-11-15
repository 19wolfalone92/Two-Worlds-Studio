"""EB01 V8 audio-conditioned performance evidence gate (metadata only).

No private WAV, portrait, prompt, frames, or signed URLs are accepted here.
A passing numeric synchronization metric is not an artistic sign-off.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from studio.eb01_v8_40s_native_motion_plan import plan

HEX64 = re.compile(r"^[a-f0-9]{64}$")


def verify_audio_conditioned_performances(evidence: Mapping[str, dict]) -> dict:
    """Require genuine six-line temporal alignment AND jaw-specific motion evidence.

    Each record is produced privately from decoded original frames and the
    canonical WAV. Correlation alone is insufficient: owner review is required.
    Fail closed on missing, nonfinite, or fabricated evidence.
    """
    schedules = {s["scene_id"]: s for s in plan()["scenes"]
                 if s["canonical_voice_sha256"] is not None}
    reasons = []
    if not isinstance(evidence, Mapping) or set(evidence) != set(schedules):
        return {"status": "BLOCKED", "reasons": ["SIX_ORIGINAL_LINES_REQUIRED"],
                "verified_lines": 0, "movie_ready": False}
    verified = 0
    for sid, scene in schedules.items():
        item = evidence[sid]
        if not isinstance(item, Mapping):
            reasons.append(f"{sid}_INVALID_EVIDENCE")
            continue
        start = scene["dialogue_start_scene_frame"]
        end = scene["dialogue_end_scene_frame"]
        numeric = ("audio_visual_offset_ms", "jaw_audio_correlation",
                   "voiced_jaw_motion_px", "silent_jaw_motion_px")
        valid_numbers = all(isinstance(item.get(k), (int, float))
                            and not isinstance(item.get(k), bool)
                            and math.isfinite(item[k]) for k in numeric)
        if (item.get("voice_sha256") != scene["canonical_voice_sha256"]
            or item.get("scene_id") != sid
            or item.get("dialogue_start_frame") != start
            or item.get("dialogue_end_frame") != end
            or item.get("decoded_scene_frames") != scene["unique_movie_frames"]
            or item.get("original_neural_frames") is not True
            or not valid_numbers):
            reasons.append(f"{sid}_SOURCE_TIMING_OR_METRICS_UNVERIFIED")
            continue
        if (abs(item["audio_visual_offset_ms"]) > 80
            or item["jaw_audio_correlation"] < 0.45
            or item["jaw_audio_correlation"] > 1.0
            or item["voiced_jaw_motion_px"] <= item["silent_jaw_motion_px"]):
            reasons.append(f"{sid}_AUDIO_CONDITIONING_NOT_DEMONSTRATED")
            continue
        # Bone jaw and human cheek/jaw require separate visual review.
        if item.get("anatomical_artistic_owner_review") != "PASS":
            reasons.append(f"{sid}_ANATOMICAL_OWNER_REVIEW_MISSING")
            continue
        verified += 1
    return {"status": "EVIDENCE_PASS_OWNER_FINAL_FILM_REVIEW_REQUIRED"
            if not reasons else "BLOCKED",
            "reasons": sorted(set(reasons)), "verified_lines": verified,
            "expected_lines": 6, "movie_ready": False, "public_media_export": False}
