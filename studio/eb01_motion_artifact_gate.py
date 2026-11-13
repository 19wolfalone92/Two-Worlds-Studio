"""Reject sudden neon artifacts in otherwise dark EB01 Wan I2V pilot frames.

Real production incident: first 46 frames preserved Kostya; last three frames
showed high-saturation stripe artifacts in the lower part of the image.
Technical all_frames_decoded=True did NOT detect this aesthetic failure.

This is an auditable CPU-only diagnostic and a SAFE optional tail-only salvage,
not an image generator or a claim that animation is artistically approved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np


class ArtifactRejected(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    data = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,avg_frame_rate",
         "-of", "json", str(path)], text=True,
        capture_output=True, check=True, timeout=30)
    streams = json.loads(data.stdout).get("streams", [])
    if len(streams) != 1:
        raise ArtifactRejected("Video requires exactly one readable video stream")
    return streams[0]


def bright_color_fraction(frame: np.ndarray) -> float:
    """Measure unnaturally vivid bright pixels in the bottom 14% of each frame."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ArtifactRejected("Expected RGB video frame")
    y0 = int(frame.shape[0] * .86)
    roi = frame[y0:].astype(np.int16)
    spread = roi.max(axis=2) - roi.min(axis=2)
    mask = (roi.max(axis=2) > 150) & (spread > 65)
    return float(np.mean(mask))


def horizontal_neon_band_fraction(frame: np.ndarray) -> float:
    """Detect broad bright RGB/cyan/magenta stripes anywhere in frame."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ArtifactRejected("Expected RGB video frame")
    rgb = frame.astype(np.int16)
    hi, lo = rgb.max(axis=2), rgb.min(axis=2)
    chroma = (hi >= 165) & ((hi - lo) >= 95)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    cyan = (g > r + 65) & (b > r + 65)
    red = (r > g + 75) & (r > b + 60)
    magenta = (r > g + 65) & (b > g + 65)
    row_fraction = np.mean(chroma & (cyan | red | magenta), axis=1)
    return float(np.mean(row_fraction >= .35))


def warm_scene_spectral_intrusion_fraction(frame: np.ndarray) -> float:
    """Unnatural blue/cyan/magenta streak pixels in EB01's warm gothic palette.

    Previous QC only looked at bottom-of-frame neon or bars spanning >=35% of
    a row, missing the actual S02 thin multicolour stripes across furniture.
    Deliberately false-positive conservative: quarantine, never auto-correct.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ArtifactRejected("Expected RGB video frame")
    rgb = frame.astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    blue = (b > 100) & (b > r + 60) & (b > g + 40)
    cyan = (b > 100) & (g > 90) & (b > r + 50) & (g > r + 40)
    magenta = (r > 100) & (b > 100) & (r > g + 55) & (b > g + 50)
    return float(np.mean(blue | cyan | magenta))


def classify(values: list[float]) -> dict:
    if len(values) < 10:
        raise ArtifactRejected("Pilot has too few frames to check continuity")
    baseline = float(np.median(values[:max(5, round(len(values) * .7))]))
    threshold = max(.028, baseline + .025, baseline * 12)
    outliers = [i for i, x in enumerate(values) if x > threshold]
    if not outliers:
        return {"status": "NO_NEON_ARTIFACT_FOUND", "safe_frame_count": len(values),
                "flagged_frames": [], "baseline": baseline, "threshold": threshold}
    start = outliers[0]
    terminal = (outliers == list(range(start, len(values))))
    if not terminal:
        return {"status": "REJECT_NONTERMINAL_NEON_ARTIFACT",
                "safe_frame_count": 0, "flagged_frames": outliers,
                "baseline": baseline, "threshold": threshold}
    if len(outliers) < 2 or (len(values) - len(outliers)) < 15:
        return {"status": "REJECT_INSUFFICIENT_UNCORRUPTED_PILOT",
                "safe_frame_count": 0, "flagged_frames": outliers,
                "baseline": baseline, "threshold": threshold}
    return {"status": "TRIM_ONLY_TERMINAL_ARTIFACTS",
            "safe_frame_count": start, "flagged_frames": outliers,
            "baseline": baseline, "threshold": threshold}


def inspect_clip(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size < 512:
        raise ArtifactRejected("Source MP4 missing or invalid")
    meta = probe(path)
    w, h = int(meta["width"]), int(meta["height"])
    fps = meta.get("avg_frame_rate")
    if w < 128 or h < 128 or w*h > 1600*1000 or fps != "24/1":
        raise ArtifactRejected("Only native 24 fps, bounded Wan pilot permitted")
    frame_bytes = w*h*3
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
               "-i", str(path), "-an", "-pix_fmt", "rgb24",
               "-f", "rawvideo", "-"]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    values = []
    horizontal_trace = []
    spectral_trace = []
    try:
        while len(values) < 400:
            raw = proc.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise ArtifactRejected("Truncated raw frame")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h,w,3))
            values.append(bright_color_fraction(frame))
            horizontal_trace.append(horizontal_neon_band_fraction(frame))
            spectral_trace.append(warm_scene_spectral_intrusion_fraction(frame))
        if len(values) >= 400:
            raise ArtifactRejected("Too many frames for bounded pilot check")
        if proc.wait(timeout=30) != 0:
            raise ArtifactRejected("ffmpeg decode failed")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    result = classify(values)
    band_frames = [i for i, v in enumerate(horizontal_trace) if v >= .012]
    if len(band_frames) >= 2:
        result["status"] = "REJECT_HORIZONTAL_NEON_BANDS_REVIEW_REQUIRED"
        result["safe_frame_count"] = 0
        result["flagged_frames"] = sorted(set(result["flagged_frames"] + band_frames))
    # Even sparse RGB streaks are visible in the real 320px S02 background.
    # Require two bad frames; one sparkle of blue can be legitimate cinema.
    spectral_baseline = float(np.median(spectral_trace[:max(5, round(len(spectral_trace)*.35))]))
    spectral_threshold = max(.006, spectral_baseline + .004)
    spectral_bad = [i for i, v in enumerate(spectral_trace) if v > spectral_threshold]
    if len(spectral_bad) >= 2:
        result["status"] = "REJECT_SPECTRAL_NEON_IN_WARM_SCENE"
        result["safe_frame_count"] = 0
        result["flagged_frames"] = sorted(set(result["flagged_frames"] + spectral_bad))
    result["spectral_intrusion_trace"] = [round(v, 5) for v in spectral_trace]
    result["spectral_intrusion_flagged_frames"] = spectral_bad
    result["spectral_intrusion_threshold"] = round(spectral_threshold, 5)
    result["horizontal_neon_band_trace"] = [round(v, 5) for v in horizontal_trace]
    result["horizontal_neon_band_flagged_frames"] = band_frames
    if meta.get("nb_frames") and int(meta["nb_frames"]) != len(values):
        raise ArtifactRejected("Decoded frame count differs from stream metadata")
    result.update({"source_video_sha256": sha256(path), "width": w,
                   "height": h, "fps": fps, "decoded_frames": len(values),
                   "frame_saturation_trace": [round(x, 5) for x in values],
                   "actor_motion_artistically_approved": False, "published": False})
    return result


def trim_bad_tail(path: Path, output: Path) -> dict:
    data = inspect_clip(path)
    if data["status"] != "TRIM_ONLY_TERMINAL_ARTIFACTS":
        raise ArtifactRejected("Only provably corrupt *terminal* video frames may be trimmed")
    if output.exists() or output.is_symlink() or path.resolve() == output.resolve():
        raise ArtifactRejected("Never overwrite existing pilot or source")
    if output.suffix.lower() != ".mp4":
        raise ArtifactRejected("Output must be a new MP4 file")
    usable = data["safe_frame_count"]
    # No color restoration, frame hallucination or fake I2V inference.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", "-nostdin", "-loglevel", "error",
         "-i", str(path), "-vf", f"trim=end_frame={usable},setpts=PTS-STARTPTS",
         "-an", "-r", "24", "-c:v", "libx264", "-preset", "slow",
         "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(output)], capture_output=True, check=True, timeout=180)
    checked = inspect_clip(output)
    if checked["decoded_frames"] != usable or checked["flagged_frames"]:
        output.unlink(missing_ok=True)
        raise ArtifactRejected("Filtered pilot still has corrupt frames")
    return {"status": "TERMINAL_ARTIFACTS_REMOVED_FOR_PRIVATE_REVIEW_ONLY",
            "source_sha256": data["source_video_sha256"],
            "result_sha256": sha256(output),
            "source_frames": data["decoded_frames"], "result_frames": usable,
            "dropped_frames": data["flagged_frames"],
            "real_i2v_source_preserved": True,
            "human_motion_approval": False, "full_episode_ready": False,
            "published": False}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--trim-artifact-tail-to", type=Path)
    args = parser.parse_args(argv)
    output = (trim_bad_tail(args.video, args.trim_artifact_tail_to)
              if args.trim_artifact_tail_to else inspect_clip(args.video))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
