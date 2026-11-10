"""Render a trusted local Blender scene as a silent portrait shot."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from studio.production_queue import private_path
from studio.release_gate import sha256
from studio.smooth_camera import encode_rgb_frames


def render_clip(source: Path, output: Path, *, first_frame: int = 1, frames: int,
                width: int = 540, fps: int = 30, samples: int = 16,
                threads: int = 4) -> dict:
    """Use offline Cycles CPU, validate every frame, then atomically commit the MP4."""
    limits = ((first_frame, 1, 9700), (frames, 1, 300), (width, 180, 1080),
              (fps, 12, 30), (samples, 4, 128), (threads, 1, 16))
    if any(type(value) is not int or not lo <= value <= hi for value, lo, hi in limits):
        raise ValueError("Blender clip settings exceed the supported CPU render bounds")
    if width % 18:
        raise ValueError("Width must be a multiple of 18 for an even 9:16 canvas")
    if source.suffix.lower() != ".blend" or not source.is_file():
        raise ValueError("A readable, trusted local .blend is required")
    if output.suffix.lower() != ".mp4" or output.is_symlink():
        raise ValueError("Choose a new private .mp4 output")
    source, output = source.resolve(), private_path(output)
    if output.exists():
        raise FileExistsError("Choose a new output path")
    blender = shutil.which("blender")
    if not blender or not shutil.which("ffmpeg"):
        raise RuntimeError("Blender 4.3+ and FFmpeg must be available on PATH")

    height, end = width * 16 // 9, first_frame + frames - 1
    fingerprint = sha256(source)
    # User-controlled paths stay in subprocess arguments. The expression only
    # contains validated integers and fixed renderer settings.
    settings = "\n".join((
        "import bpy",
        "scene = bpy.context.scene",
        "if scene.camera is None: raise ValueError('Select a scene camera first')",
        "r = scene.render",
        "r.engine = 'CYCLES'",
        "scene.cycles.device = 'CPU'",
        f"scene.cycles.samples = {samples}",
        f"r.resolution_x, r.resolution_y, r.resolution_percentage = {width}, {height}, 100",
        f"r.fps, r.fps_base = {fps}, 1.0",
        "r.pixel_aspect_x, r.pixel_aspect_y = 1, 1",
        "r.use_border = r.film_transparent = False",
        "r.use_sequencer = False",
        "r.image_settings.file_format = 'PNG'",
        "r.image_settings.color_mode = 'RGB'",
        "scene.frame_step = 1",
        f"r.threads_mode, r.threads = 'FIXED', {threads}",
    ))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tw-blender-", dir=output.parent) as directory:
        work = Path(directory)
        prefix = work / "shot_####"
        log = work / "blender.log"
        with log.open("wb") as diagnostics:
            subprocess.run([
                blender, "--background", "--factory-startup", "--offline-mode", "--disable-autoexec",
                str(source), "--python-exit-code", "10", "--python-expr", settings,
                "-o", str(prefix), "-x", "1", "-s", str(first_frame), "-e", str(end), "-a",
            ], cwd=source.parent, stdout=diagnostics, stderr=subprocess.STDOUT, check=True, timeout=1800)
        filenames = [work / f"shot_{frame:04}.png" for frame in range(first_frame, end + 1)]
        if any(not path.is_file() for path in filenames):
            raise RuntimeError("Blender did not render the requested frame range. Check scene settings.")
        if sha256(source) != fingerprint:
            raise RuntimeError("The Blender scene changed during rendering")

        def images():
            for path in filenames:
                with Image.open(path) as frame:
                    yield frame.convert("RGB")

        temporary = work / "complete.mp4"
        encode_rgb_frames(temporary, width, height, fps, frames, images())
        digest = sha256(temporary)
        os.link(temporary, output)

    return {"status": "rendered", "backend": "blender_cycles_cpu",
            "first_frame": first_frame, "frames": frames, "fps": fps,
            "width": width, "height": height, "scene_sha256": fingerprint,
            "clip_sha256": digest, "published": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-frame", type=int, default=1)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--width", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    try:
        result = render_clip(**vars(parser.parse_args()))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Blender clip blocked: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
