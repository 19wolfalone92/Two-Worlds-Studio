"""Smooth deterministic Ken Burns motion with sub-pixel bicubic sampling.

Each frame is sampled from the same overscanned still using a floating-point
affine inverse transform, avoiding integer crop stepping in FFmpeg zoompan.
No neural generation, network, token, publishing or GPU required.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

CAMERAS = frozenset({"push_in", "pull_out", "pan_left", "pan_right", "static"})


def camera_pose(frame: int, frames: int, camera: str, strength: float = 0.45) -> tuple[float, float, float]:
    """(zoom, normalized pan_x, normalized pan_y), with zero start/end velocity."""
    if camera not in CAMERAS:
        raise ValueError(f"Unsupported camera movement: {camera}")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Camera strength must be within 0..1")
    t = max(0.0, min(1.0, frame / max(1, frames - 1)))
    eased = (1.0 - math.cos(math.pi * t)) / 2.0
    amplitude = min(0.024 * strength / 0.45, 0.05) if strength else 0.0
    pan = min(0.12 * strength / 0.45, 0.24) if strength else 0.0
    if camera == "push_in":
        return (1.008 + amplitude * eased, 0.0, 0.0)
    if camera == "pull_out":
        return (1.008 + amplitude * (1 - eased), 0.0, 0.0)
    if camera == "pan_left":
        return (1.025, pan * (1 - 2 * eased), 0.0)
    if camera == "pan_right":
        return (1.025, pan * (2 * eased - 1), 0.0)
    return (1.0, 0.0, 0.0)


def render_still(image_path: Path, target_path: Path, width: int, height: int, fps: int,
                 frames: int, camera: str, strength: float = 0.45) -> None:
    """Export constant-FPS H.264 using independently bicubic-sampled frames."""
    if frames < 1 or fps < 1 or width % 2 or height % 2 or min(width, height) < 64:
        raise ValueError("Invalid frame count, resolution or FPS")
    # Validate before launching FFmpeg, even if render_still is called directly.
    camera_pose(0, frames, camera, strength)
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        iw, ih = math.ceil(width * 1.12), math.ceil(height * 1.12)
        src = ImageOps.fit(img, (iw, ih), method=Image.Resampling.LANCZOS)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as error_file:
        proc = subprocess.Popen([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-frames:v", str(frames),
            "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", str(fps), str(target_path)
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_file)
        try:
            for i in range(frames):
                zoom, px, py = camera_pose(i, frames, camera, strength)
                scale = 1.0 / zoom
                cx = iw / 2 + px * (iw - width) / 2
                cy = ih / 2 + py * (ih - height) / 2
                ox = cx - (width / 2) * scale
                oy = cy - (height / 2) * scale
                frame = src.transform(
                    (width, height), Image.Transform.AFFINE,
                    (scale, 0.0, ox, 0.0, scale, oy),
                    resample=Image.Resampling.BICUBIC)
                assert proc.stdin is not None
                proc.stdin.write(frame.tobytes())
        except BaseException:
            proc.kill()
            proc.wait()
            target_path.unlink(missing_ok=True)
            raise
        finally:
            if proc.stdin:
                proc.stdin.close()
        try:
            code = proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            target_path.unlink(missing_ok=True)
            raise RuntimeError("FFmpeg camera render exceeded timeout")
        if code:
            error_file.seek(0)
            target_path.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg still rendering failed: {error_file.read(3000)!r}")
