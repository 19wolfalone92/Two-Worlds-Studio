"""Fail-closed Studio native-video job state and proof checks.

Kaggle queueing is not a neural-inference failure. Never classify the GitHub
runner's observation deadline as GPU failure and never resubmit a queued job.
This module never opens original user media and never prints private MP4 bytes.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

KERNEL = "wolf19loner92/studio-native-man-wave-81a7215251acc6b"
SOURCE_SHA256 = "379d772afa4b7f1448b0881d84631776329ecf1d670ac8da9b5701f2ea5edc72"
ALLOWED_STATES = frozenset({
    "QUEUED", "RUNNING", "COMPLETE", "ERROR", "CANCEL", "CANCELLED", "CANCEL_ACKNOWLEDGED"
})
PENDING = frozenset({"QUEUED", "RUNNING"})
FAILURE = frozenset({"ERROR", "CANCEL", "CANCELLED", "CANCEL_ACKNOWLEDGED"})
RECEIPT_NAME = "STUDIO_WAVE_TRUE_NATIVE_RECEIPT.json"
VIDEO_NAME = "STUDIO_WAVE_TRUE_NATIVE_65F_24FPS.mp4"

class JobStateError(RuntimeError):
    """Unverified remote state or an incomplete/invalid result."""

def parse_state(value: str) -> str:
    if not isinstance(value, str):
        raise JobStateError("STATUS_NOT_TEXT")
    matches = re.findall(r"KernelWorkerStatus\.([A-Z_]+)", value)
    if len(matches) != 1 or matches[0] not in ALLOWED_STATES:
        raise JobStateError("AMBIGUOUS_OR_UNKNOWN_KAGGLE_STATE")
    return matches[0]

def classify(state: str) -> dict:
    if state not in ALLOWED_STATES:
        raise JobStateError("UNSUPPORTED_KAGGLE_STATE")
    if state in PENDING:
        return {
            "status": "WAITING_GPU" if state == "QUEUED" else "GPU_RUNNING",
            "model_failure_confirmed": False,
            "real_mp4_accepted": False,
            "safe_to_requeue": False,
            "requires_later_readback": True,
        }
    if state in FAILURE:
        return {
            "status": "KAGGLE_TERMINAL_FAILURE",
            "model_failure_confirmed": False, # inspect private GPU logs to locate failure
            "real_mp4_accepted": False,
            "safe_to_requeue": False,
            "requires_later_readback": False,
        }
    return {
        "status": "COMPLETE_REQUIRES_INDEPENDENT_VIDEO_VERIFICATION",
        "model_failure_confirmed": False,
        "real_mp4_accepted": False,
        "safe_to_requeue": False,
        "requires_later_readback": False,
    }

def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def verify_actual_native_mp4(output_dir: Path) -> dict:
    """Full original-neural output acceptance: not a cosmetic/static remaster."""
    folder=Path(output_dir)
    receipt_path=folder/RECEIPT_NAME
    video_path=folder/VIDEO_NAME
    if not receipt_path.is_file() or not video_path.is_file():
        raise JobStateError("Genuine GPU MP4 or receipt missing")
    if receipt_path.stat().st_size > 100_000 or not 8000 < video_path.stat().st_size < 145_000_000:
        raise JobStateError("Native media/receipt has unrealistic size")
    proof=json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "schema": 1,
        "status": "NATIVE_NEURAL_VIDEO_GENERATED_ARTISTIC_REVIEW_REQUIRED",
        "model": "Lightricks/LTX-Video-2B-0.9.6-distilled",
        "source_reference_sha256": SOURCE_SHA256,
        "source": "original_user_reference_jpeg_only_not_previous_video_frames",
        "actual_native_gpu_inference": True,
        "output_frames": 65,
        "output_width": 576,
        "output_height": 1024,
        "output_fps": 24,
        "extra_interpolated_frames": 0,
        "private_media_only": True,
        "published": False,
        "human_artistic_approved": False,
        "anatomical_continuity_human_review_required": True,
    }
    if any(proof.get(k) != v for k,v in expected.items()):
        raise JobStateError("Genuine native model receipt policy mismatch")
    if proof.get("qa_color_all_segments") is not True:
        raise JobStateError("Native model detected color artifacts; owner must review raw source")
    if not re.fullmatch(r"[a-f0-9]{40}",str(proof.get("source_git_sha",""))):
        raise JobStateError("Missing original source commit provenance")
    expected_sha=_sha256(video_path)
    if proof.get("output_sha256") != expected_sha:
        raise JobStateError("MP4 byte-level source provenance mismatch")
    segments=proof.get("original_neural_segments")
    if not isinstance(segments,list) or len(segments)!=2:
        raise JobStateError("Two REAL neural segments required; no still-only fallback")
    for index,record in enumerate(segments):
        if (record.get("index")!=index or record.get("frames")!=33
                or record.get("fps")!=24 or record.get("width")!=576
                or record.get("height")!=1024
                or not re.fullmatch(r"[a-f0-9]{64}",str(record.get("sha256","")))):
            raise JobStateError("Native neural segment proof incomplete")
        elapsed=record.get("model_seconds")
        if elapsed is None:
            # Prior private Kaggle controller used the wrong EB01 telemetry
            # prefix. The first LTX clip still exists, was independently
            # decoded and has a fixed real model SHA and output-ready stage.
            if not (index==0 and
                    record.get("recovered_from_prior_private_kernel") is True and
                    record.get("native_output_stage_attested") is True and
                    record.get("sha256") ==
                    "0e2f9c33e172ef48d8bf1761c06768eea661387b954830430071a28074803885"):
                raise JobStateError("Recovered segment must have exact real model provenance")
        elif (not isinstance(elapsed,(int,float)) or not 0<elapsed<=3600):
            raise JobStateError("Original native segment timing invalid")
        if record.get("color_qa") != "NO_NEON_ARTIFACT_FOUND":
            raise JobStateError("Native source contains color/artifact QA flags")
        motion=record.get("numerical_motion")
        if (not isinstance(motion,dict) or
            motion.get("decoded_native_frames") != 33 or
            not isinstance(motion.get("mean_frame_luma_delta"),(int,float)) or
            motion["mean_frame_luma_delta"] <= 0.18 or
            not isinstance(motion.get("nonidentical_motion_frames"),int) or
            motion["nonidentical_motion_frames"] < 4):
            raise JobStateError("Native segment is frozen or has insufficient independent frame motion")
        original=folder/("STUDIO_WAVE_REAL_SEGMENT_"+str(index)+".mp4")
        if not original.is_file() or _sha256(original)!=record["sha256"]:
            raise JobStateError("Original full-frame neural GPU segment absent or SHA mismatch")
    cmd=["ffprobe","-v","error","-count_frames","-select_streams","v:0",
         "-show_entries","stream=width,height,avg_frame_rate,nb_read_frames",
         "-of","json",str(video_path)]
    try:
        meta=json.loads(subprocess.check_output(cmd,timeout=45))
        stream=meta["streams"][0]
        dims=(stream.get("width"),stream.get("height"),
              stream.get("avg_frame_rate"),str(stream.get("nb_read_frames")))
        if dims!=(576,1024,"24/1","65"):
            raise JobStateError("Decoded video format/frame count does not match receipt")
        subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(video_path),
                        "-f","null","-"],check=True,capture_output=True,timeout=120)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired,
            ValueError,IndexError,KeyError) as exc:
        raise JobStateError("Full native video decode failed") from exc
    return {
        "status": "TRUE_NEURAL_MP4_TECHNICALLY_VERIFIED_OWNER_REVIEW_REQUIRED",
        "actual_native_gpu_inference": True,
        "frames":65,"fps":24,"size":[576,1024],
        "sha256":expected_sha,
        "human_artistic_approved":False,
        "finished_movie_quality_guaranteed":False,
        "published":False,
    }
