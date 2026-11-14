"""CPU-only streaming RGB forensic scan for long EB01 review clips.

This never approves artistic quality or modifies source footage.
Usage: python tools/eb01_streaming_rgb_forensics.py input.mp4 --output report.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from studio.eb01_motion_artifact_gate import (
    ArtifactRejected, narrow_neon_intrusion_fraction, probe, sha256,
)

THRESHOLD = 0.0015


def scan(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size < 512:
        raise ArtifactRejected("Missing or invalid source video")
    meta = probe(path)
    width, height = int(meta["width"]), int(meta["height"])
    if not (128 <= width <= 1920 and 128 <= height <= 1920):
        raise ArtifactRejected("Unsupported dimensions")
    frame_bytes = width * height * 3
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
               "-i", str(path), "-map", "0:v:0", "-vsync", "0",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    flagged = []
    count = 0
    try:
        while True:
            # Pipes may return short reads even when the decoder is healthy.
            # Accumulate a complete frame before deciding it is truncated.
            chunks = []
            remaining = frame_bytes
            while remaining:
                part = proc.stdout.read(remaining)
                if not part:
                    break
                chunks.append(part)
                remaining -= len(part)
            if remaining == frame_bytes:
                break
            if remaining:
                raise ArtifactRejected(f"Truncated RGB frame {count}: missing {remaining} bytes")
            raw = b"".join(chunks)
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            if narrow_neon_intrusion_fraction(frame) > THRESHOLD:
                flagged.append(count)
            count += 1
        if proc.wait(timeout=60) != 0:
            raise ArtifactRejected("ffmpeg decoder failed")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdout:
            proc.stdout.close()
    if count == 0:
        raise ArtifactRejected("No decoded frames")
    if meta.get("nb_frames") and int(meta["nb_frames"]) != count:
        raise ArtifactRejected("Decoded frame count mismatch")
    return {
        "source_sha256": sha256(path),
        "decoded_frames": count,
        "suspect_frame_indices_zero_based": flagged,
        "suspect_count": len(flagged),
        "threshold": THRESHOLD,
        "rgb_status": "REVIEW_REQUIRED" if flagged else "NO_RGB_SUSPECTS_DETECTED",
        "artistic_status": "NOT_APPROVED",
        "lipsync_status": "NOT_VERIFIED",
        "publication_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ArtifactRejected("Refusing to overwrite an existing report")
    result = scan(args.video)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rgb_status": result["rgb_status"], "suspect_count": result["suspect_count"]}))


if __name__ == "__main__":
    main()
