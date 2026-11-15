"""Read-only numerical review of the ACTUAL private completed LTX shot, not a fake film.

Only safe compact metrics are printed/saved in a PUBLIC GitHub Actions repo.
Private MP4 and original portraits/voice NEVER leave temporary GitHub runner.
No GPU push, no inference, no publication.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from fractions import Fraction

KERNEL = "wolf19loner92/eb01-ltx2b-pilot33-c28a516c464e"


def error_family_only(raw: str) -> str:
    """Classify Kaggle CLI failures WITHOUT logging server error body."""
    value = raw.lower()
    if any(part in value for part in ("429", "too many requests", "rate limit", "quota exceeded")):
        return "HTTP_429_KAGGLE_RATE_LIMIT_PAUSE_REQUESTS"
    if any(part in value for part in ("401", "unauthorized", "invalid token", "expired token")):
        return "KAGGLE_TOKEN_UNAUTHORIZED"
    if any(part in value for part in ("403", "forbidden", "permission denied", "permissionerror")):
        return "KAGGLE_DATASET_OR_KERNEL_ACL_DENIED"
    if any(part in value for part in ("404", "not found", "notfound")):
        return "KAGGLE_PRIVATE_KERNEL_OR_OUTPUT_NOT_FOUND"
    if any(part in value for part in ("503", "502", "504", "500", "internal server error")):
        return "KAGGLE_API_SERVER_UNAVAILABLE"
    if any(part in value for part in ("tls", "ssl", "connection", "timeout", "timed out")):
        return "KAGGLE_NETWORK_OR_TLS_FAILURE"
    return "KAGGLE_NONZERO_EXIT_UNCLASSIFIED"


def process(args: list[str], timeout: int = 180) -> tuple[int, str]:
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=timeout, check=False)
        # Only return a predefined ERROR CLASS, never private paths, prompt
        # fragments, original Kaggle account identifiers, or service tokens.
        return p.returncode, ("OK" if p.returncode == 0 else
            error_family_only((p.stderr or "") + "\n" + (p.stdout or "")))
    except (OSError, subprocess.TimeoutExpired):
        return 99, "CLIENT_TIMEOUT_OR_UNAVAILABLE"


def gray_motion(path: Path) -> dict:
    import numpy as np
    width, height = 64, 64
    output = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
         "-i", str(path), "-an", "-vf", f"scale={width}:{height}:flags=area",
         "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False, timeout=90)
    if output.returncode or len(output.stdout) % (width * height):
        raise ValueError("Native clip motion decode failed")
    frames = np.frombuffer(output.stdout, dtype=np.uint8).reshape(-1, height, width)
    if len(frames) != 33:
        raise ValueError("Native 33 frame clock not present")
    diffs = [float(np.abs(frames[i].astype("int16") -
                          frames[i-1].astype("int16")).mean())
             for i in range(1, len(frames))]
    return {
        "decoded_native_frames": int(len(frames)),
        "mean_frame_luma_delta": round(float(np.mean(diffs)), 4),
        "max_frame_luma_delta": round(float(np.max(diffs)), 4),
        "min_frame_luma_delta": round(float(np.min(diffs)), 4),
        "consecutive_near_duplicate_frame_pairs": int(sum(x < 0.3 for x in diffs)),
        "nonidentical_motion_frames": int(sum(x >= 0.3 for x in diffs)),
        "note": "A nonzero numeric delta is NOT proof of natural acting or lip sync",
    }


def check(root: Path) -> dict:
    from studio.eb01_motion_artifact_gate import inspect_clip
    from studio.comfy_neural import _probe

    movie = root / "S02_PILOT33.mp4"
    receipt = root / "EB01_S02_LTX_PILOT33_JOB.json"
    details = {
        "schema": 1, "source_kaggle_kernel": KERNEL,
        "model_gpu_ran": None, "only_original_pilot": True,
        "new_gpu_job_launched": False, "published": False,
        "full_movie_ready": False, "owner_artistic_approved": False,
        "private_media_exported_in_github_artifacts": False,
        "actual_mp4_exists_in_private_kaggle_output": movie.is_file(),
        "actual_receipt_exists": receipt.is_file(),
    }
    if receipt.is_file():
        try:
            data = json.loads(receipt.read_text(encoding="utf-8"))
            details["authentic_gpu_receipt"] = {
                key: data.get(key) for key in
                ("schema", "shot", "success", "status", "error_type")
            }
            # The LTX GPU worker creates a 'success:false' receipt whenever
            # numeric video/frame/color QA blocks, even after real clip output.
        except (UnicodeError, ValueError):
            details["receipt_unreadable"] = True
    if not movie.is_file():
        details["status"] = "REAL_KAGGLE_OUTPUT_MISSING_GPU_CLIP"
        return details
    details["native_mp4_byte_size"] = movie.stat().st_size
    details["native_mp4_sha256"] = hashlib.sha256(movie.read_bytes()).hexdigest()
    try:
        meta = _probe(movie)
        details["native_ffprobe"] = meta
        details["model_gpu_ran"] = meta.get("frames") == 33
        details["frames_match_required_33"] = meta.get("frames") == 33
        details["frame_rate_is_native_24"] = Fraction(meta.get("fps", "0")) == 24
        details["actual_geometry_is_320x576"] = (
            meta.get("width") == 320 and meta.get("height") == 576
        )
    except Exception as exc:
        details["native_ffprobe_error_type"] = type(exc).__name__
    try:
        qa = inspect_clip(movie)
        details["raw_color_quality_gate"] = {
            "status": qa.get("status"),
            "decoded_frames": qa.get("decoded_frames"),
            "safe_frame_count": qa.get("safe_frame_count"),
            "flagged_frames": qa.get("flagged_frames", [])[:100],
            "spectral_intrusion_flagged_frames": qa.get(
                "spectral_intrusion_flagged_frames", [])[:100],
            "horizontal_neon_band_flagged_frames": qa.get(
                "horizontal_neon_band_flagged_frames", [])[:100],
            "narrow_neon_suspected_frames": qa.get(
                "narrow_neon_suspected_frames", [])[:100],
            "spectral_threshold": qa.get("spectral_intrusion_threshold"),
            "suspect_pixel_fractions": [
                {"index": i, "fraction": qa["frame_saturation_trace"][i]}
                for i in qa.get("flagged_frames", [])[:8]
            ],
            "actor_motion_artistically_approved": False,
        }
    except Exception as exc:
        details["frame_color_probe_error_type"] = type(exc).__name__
    try:
        details["genuine_frame_motion_metrics"] = gray_motion(movie)
    except Exception as exc:
        details["motion_probe_error_type"] = type(exc).__name__
    outcome = (
        details.get("frames_match_required_33") is True
        and details.get("frame_rate_is_native_24") is True
        and details.get("actual_geometry_is_320x576") is True
        and details.get("raw_color_quality_gate", {}).get("status")
        == "NO_NEON_ARTIFACT_FOUND"
    )
    details["status"] = (
        "REAL_GPU_PILOT_PASS_CPU_QUALITY_NEEDS_HUMAN_ARTISTIC_REVIEW"
        if outcome else "REAL_GPU_PILOT_NUMERIC_QUALITY_GATE_BLOCKED"
    )
    # A 33 frame 320x576 clip is never the 40+ second cinematic release.
    return details


def main() -> int:
    report = {"schema":1, "kernel":KERNEL, "published":False,
              "private_originals_exported":False,"gpu_restarted":False,
              "full_movie_ready":False}
    with tempfile.TemporaryDirectory(prefix="eb01-private-gpu-") as folder:
        root = Path(folder)
        code, info = process(
            ["kaggle", "kernels", "output", KERNEL,
             "-p", str(root), "--force"], timeout=240)
        report["kaggle_read_only_output_download"] = info
        report["output_download_ok"] = code == 0
        if code == 0:
            report.update(check(root))
        else:
            report["status"] = "KAGGLE_PRIVATE_OUTPUT_API_UNAVAILABLE"
    path=Path(os.environ["RUNNER_TEMP"])/"EB01_REAL_GPU_NUMERIC_FRAME_QA_ONLY.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n",
                    encoding="utf-8")
    print("EB01_REAL_GPU_NATIVE_PRIVATE_CLIP_QA", json.dumps(report, ensure_ascii=False,
        sort_keys=True))
    return 0 if report.get("output_download_ok") else 2


if __name__=="__main__":
    raise SystemExit(main())
