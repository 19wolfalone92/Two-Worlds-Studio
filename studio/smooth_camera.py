"""Smooth deterministic Ken Burns motion with sub-pixel bicubic sampling.

Each frame is sampled from the same overscanned still using a floating-point
affine inverse transform, avoiding integer crop stepping in FFmpeg zoompan.
No neural generation, network, token, publishing or GPU required.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageOps

CAMERAS = frozenset({"push_in", "pull_out", "pan_left", "pan_right", "static"})


def validate_framing(framing: dict | None) -> dict | None:
    """Authored focus points, in EXIF-oriented source and screen coordinates."""
    if framing is None:
        return None
    if not isinstance(framing, dict) or set(framing) - {"source_focus", "screen_focus"}:
        raise ValueError("Framing supports only source_focus and screen_focus")
    result = {}
    for key, default in (("source_focus", None), ("screen_focus", [.46, .43])):
        point = framing.get(key, default)
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       or not 0 <= v <= 1 for v in point)):
            raise ValueError(f"{key} must contain two finite coordinates in 0..1")
        result[key] = list(point)
    return result


def ease_camera(progress: float) -> float:
    """Start and stop with zero velocity and acceleration."""
    t = min(1., max(0., progress))
    return t * t * t * (t * (6 * t - 15) + 10)


def focal_transform(frame: int, frames: int, camera: str, strength: float,
                    source_size: tuple[int, int], canvas: tuple[int, int],
                    framing: dict) -> tuple[float, float, float, float, float, float]:
    """Smooth affine path whose complete viewport stays inside the source.

    Interpolate the two already-bounded viewports, avoiding a hard-clamp kink
    half way through a pan. A focus point is author direction, not automatic
    face detection or proof that the composition was visually reviewed.
    """
    config = validate_framing(framing)
    iw, ih = source_size
    w, h = canvas
    if min(iw, ih, w, h) <= 0 or iw < w or ih < h:
        raise ValueError("Focal camera needs a covering source image")
    sx, sy = config["source_focus"]
    tx, ty = config["screen_focus"]

    def endpoint(index):
        zoom, px, py = camera_pose(index, frames, camera, strength)
        scale = 1 / zoom
        dx = px * min(iw - w * scale, w * .20) / 2
        dy = py * min(ih - h * scale, h * .20) / 2
        ox = min(iw - w * scale, max(0., sx * iw - tx * w * scale + dx))
        oy = min(ih - h * scale, max(0., sy * ih - ty * h * scale + dy))
        return scale, ox, oy

    first = endpoint(0)
    last = endpoint(max(0, frames - 1))
    eased = ease_camera(frame / max(1, frames - 1))
    scale, ox, oy = (a + (b - a) * eased for a, b in zip(first, last))
    return scale, 0., ox, 0., scale, oy


def camera_pose(frame: int, frames: int, camera: str, strength: float = 0.45) -> tuple[float, float, float]:
    """(zoom, normalized pan_x, normalized pan_y), with zero start/end velocity."""
    if camera not in CAMERAS:
        raise ValueError(f"Unsupported camera movement: {camera}")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Camera strength must be within 0..1")
    eased = ease_camera(frame / max(1, frames - 1))
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
                 frames: int, camera: str, strength: float = 0.45,
                 *, framing: dict | None = None) -> None:
    """Export constant-FPS H.264 using independently bicubic-sampled frames."""
    if frames < 1 or fps < 1 or width % 2 or height % 2 or min(width, height) < 64:
        raise ValueError("Invalid frame count, resolution or FPS")
    # Validate before launching FFmpeg, even if render_still is called directly.
    camera_pose(0, frames, camera, strength)
    framing = validate_framing(framing)
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        if framing:
            # Retain the full source instead of discarding an off-center face
            # or story prop before the camera has a chance to frame it.
            factor = max(width / img.width, height / img.height) * 1.12
            iw, ih = math.ceil(img.width * factor), math.ceil(img.height * factor)
            src = img.resize((iw, ih), resample=Image.Resampling.LANCZOS)
        else:
            iw, ih = math.ceil(width * 1.12), math.ceil(height * 1.12)
            src = ImageOps.fit(img, (iw, ih), method=Image.Resampling.LANCZOS)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if camera == 'static' or strength == 0:
        # Clone the same bicubic composition inside FFmpeg. Sampling it again
        # for every frame wastes CPU and cannot add movement to a held shot.
        zoom,px,py = camera_pose(0,frames,camera,strength)
        scale=1/zoom
        transform = (focal_transform(0, frames, camera, strength, (iw, ih),
                                    (width, height), framing) if framing else
                     (scale,0.,iw/2+px*(iw-width)/2-width/2*scale,
                      0.,scale,ih/2+py*(ih-height)/2-height/2*scale))
        frame = src.transform((width,height),Image.Transform.AFFINE, transform,
                    resample=Image.Resampling.BICUBIC)
        with tempfile.TemporaryDirectory(prefix='tw-static-') as temp:
            picture=Path(temp)/'frame.png';frame.save(picture)
            result=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
                '-framerate',str(fps),'-i',str(picture),'-vf',f'tpad=stop_mode=clone:stop={frames-1}',
                '-an','-frames:v',str(frames),'-c:v','libx264','-threads','2','-preset','veryfast',
                '-crf','18','-pix_fmt','yuv420p','-r',str(fps),str(target_path)],
                stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=120)
            if result.returncode:
                target_path.unlink(missing_ok=True)
                raise RuntimeError(f'FFmpeg static export failed: {result.stderr[-2000:]!r}')
        return
    def images():
        for i in range(frames):
            zoom, px, py = camera_pose(i, frames, camera, strength)
            scale = 1.0 / zoom
            cx = iw / 2 + px * (iw - width) / 2
            cy = ih / 2 + py * (ih - height) / 2
            ox = cx - (width / 2) * scale
            oy = cy - (height / 2) * scale
            transform = (focal_transform(i, frames, camera, strength, (iw, ih),
                                         (width, height), framing) if framing else
                         (scale, 0.0, ox, 0.0, scale, oy))
            yield src.transform((width, height), Image.Transform.AFFINE, transform,
                                resample=Image.Resampling.BICUBIC)

    encode_rgb_frames(target_path, width, height, fps, frames, images())


def encode_rgb_frames(target_path: Path, width: int, height: int, fps: int,
                      frames: int, images: Iterable[Image.Image]) -> None:
    """Stream rendered frames without buffering a complete shot in RAM."""
    with tempfile.TemporaryFile() as error_file:
        proc = subprocess.Popen([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-frames:v", str(frames),
            "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", str(fps), str(target_path)
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_file)
        try:
            assert proc.stdin is not None
            count = 0
            for count, frame in enumerate(images, 1):
                if count > frames or frame.mode != "RGB" or frame.size != (width, height):
                    raise ValueError("Rendered frame does not match the target clock or canvas")
                proc.stdin.write(frame.tobytes())
            if count != frames:
                raise ValueError("Rendered shot does not contain its full frame count")
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
