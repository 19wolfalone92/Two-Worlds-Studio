"""CPU-only temporal pixel diagnostics for the isolated Freevideo experiment.

It cannot validate character identity, skeletal anatomy, neural origin or art.
No generated video, reference or review is ever published or altered.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

SIZE = (96, 160)
MAX_FRAMES = 241
MAX_INPUT_BYTES = 512 * 1024 * 1024

class VideoRejected(ValueError):
    """MP4 is malformed, unreadable or violates bounded inputs."""

def _execute(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    try:
        p = subprocess.run(command, check=False, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VideoRejected(f"Video decoder unavailable or timed out: {type(exc).__name__}") from exc
    if p.returncode != 0:
        raise VideoRejected("Video decoder rejected the file")
    return p

def audit_video(video_path: str | Path) -> dict:
    path = Path(video_path).expanduser().absolute()
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".mp4":
        raise VideoRejected("Expected an existing regular MP4, not a symlink")
    if not 4096 <= path.stat().st_size <= MAX_INPUT_BYTES:
        raise VideoRejected("Video file size is outside permitted bounds")
    probe = _execute(["ffprobe", "-v", "error", "-select_streams", "v:0",
                      "-count_frames", "-show_entries",
                      "stream=width,height,avg_frame_rate,nb_read_frames",
                      "-of", "json", str(path)])
    try:
        streams = json.loads(probe.stdout).get("streams", [])
        if len(streams) != 1:
            raise VideoRejected("Expected one video stream")
        track = streams[0]
        frames = int(track["nb_read_frames"])
        width, height = int(track["width"]), int(track["height"])
        fps = Fraction(track["avg_frame_rate"])
        if not 9 <= frames <= MAX_FRAMES or width < 128 or height < 128 or not 1 <= fps <= 60:
            raise VideoRejected("Unsupported frame clock or geometry")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise VideoRejected("Invalid decoded metadata") from exc
    w, h = SIZE
    # Strict decoding is evidence of decode correctness, not model provenance.
    raw = _execute(["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror",
                    "-err_detect", "explode", "-threads", "1", "-i", str(path),
                    "-map", "0:v:0", "-vf", f"scale={w}:{h}:flags=area,format=gray",
                    "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"])
    if len(raw.stdout) != frames * w * h:
        raise VideoRejected("Decoded frames differ from ffprobe frame count")
    arr = np.frombuffer(raw.stdout, dtype=np.uint8).reshape(frames, h, w).astype(np.int16)
    diffs = np.abs(np.diff(arr, axis=0)).mean(axis=(1, 2)) / 255.0
    near_duplicates = int(np.count_nonzero(diffs < .001))
    median = float(np.median(diffs))
    spikes = int(np.count_nonzero(diffs > max(.14, 4 * median)))
    flags = []
    if near_duplicates / (frames - 1) >= .5:
        flags.append("POSSIBLE_FROZEN_OR_REPEATED_FRAMES")
    if spikes:
        flags.append("ABRUPT_PIXEL_CHANGE_REVIEW_REQUIRED")
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return {
        "schema": 1, "status": "CPU_PIXEL_METRICS_ONLY", "artistic_quality": "NOT_VERIFIED",
        "neural_inference": "NOT_VERIFIED", "model_identity": "NOT_VERIFIED",
        "video_sha256": sha.hexdigest(), "width": width, "height": height,
        "frames": frames, "fps": str(fps), "analysis_size": [w, h],
        "adjacent_mean_absolute_difference_median": round(median, 6),
        "adjacent_mean_absolute_difference_p95": round(float(np.percentile(diffs, 95)), 6),
        "near_duplicate_pairs": near_duplicates, "abrupt_pixel_change_pairs": spikes,
        "flags": flags, "limitations": [
            "Pixel differences cannot distinguish camera motion from actor motion.",
            "No face, hands, or skeletal-geometry identity analysis is performed.",
            "Temporal pixel metrics are not independent artistic approval."
        ],
    }

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CPU-only video diagnostics; no artistic approval")
    parser.add_argument("video", help="Existing private MP4")
    args = parser.parse_args(argv)
    try:
        result, code = audit_video(args.video), 0
    except VideoRejected as exc:
        result, code = {"status": "BLOCKED", "reason": str(exc),
                        "artistic_quality": "NOT_VERIFIED"}, 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code

if __name__ == "__main__":
    raise SystemExit(main())
