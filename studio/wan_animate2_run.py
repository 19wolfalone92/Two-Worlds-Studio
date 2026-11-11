"""Execute the reviewed Wan-Animate-2 distilled upstream on a private multi-GPU node.

Uses the ORIGINAL official inference code and user-installed checkpoint;
never vendors model weights, submits cloud jobs, makes paid calls, publishes
media or pretends that T4 has the resources of two A800 devices.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from studio.comfy_neural import NeuralBlocked, _probe
from studio.release_gate import sha256
from studio.video_ai_plan import _reference

MODEL = "Wan-AI/Wan2.2-Animate-2-14B"
UPSTREAM = "Wan-Video/Wan-Animate-2"
CHECKPOINT = "wan_animate_2_bf16_distillation.safetensors"


def _video(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,duration",
        "-of", "json", str(path)
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("Driver must contain one real video stream")
    from fractions import Fraction
    s = streams[0]
    fps = float(Fraction(s["avg_frame_rate"]))
    secs = float(s["duration"])
    if not 8 <= fps <= 120 or not .5 <= secs <= 60 or int(s["width"]) < 128 or int(s["height"]) < 128:
        raise ValueError("Invalid driving video metadata")
    return {"fps": fps, "seconds": secs, "width": int(s["width"]), "height": int(s["height"])}


def plan(*, runtime: Path, reference: Path, driver: Path, output: Path, prompt: str,
         seed: int = 42, width: int = 480, height: int = 848, frames: int = 81) -> dict:
    runtime, reference, driver, output = (
        Path(runtime).resolve(), Path(reference).resolve(),
        Path(driver).resolve(), Path(output).absolute())
    picture = _reference(reference)
    if not shutil.which("ffprobe"):
        raise NeuralBlocked("ffprobe is required")
    if (not driver.is_file() or driver.suffix.lower() not in {".mp4", ".mov"}
            or not isinstance(prompt, str) or not 12 <= len(prompt.strip()) <= 1800
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(width) is not int or type(height) is not int
            or width % 16 or height % 16 or width < 320 or height < 544
            or abs(width / height - 9 / 16) > .025
            or type(frames) is not int or not 17 <= frames <= 241 or (frames - 1) % 4):
        raise ValueError("Invalid official Wan-Animate-2 distilled shot geometry and inputs")
    if output.suffix.lower() != ".mp4" or output.exists() or output.is_symlink():
        raise ValueError("Use a new private MP4 destination")
    if output.with_suffix(".receipt.json").exists():
        raise FileExistsError("Wan-Animate-2 receipt already exists")
    entry = runtime / "infer" / "wan_animate_2_demo.py"
    cfg = runtime / "infer" / "wan_animate_2_distillation.yaml"
    if not entry.is_file() or not cfg.is_file():
        raise ValueError("Clone the official Wan-Animate-2 inference runtime and install its requirements")
    driver_meta = _video(driver)
    if driver_meta["seconds"] + .001 < frames / 24:
        raise ValueError("Driving performance is shorter than the intended neural take")
    return {
        "status": "PLANNED_LOCAL_GPU_ONLY", "engine": "neural_animate2",
        "model": MODEL, "upstream": UPSTREAM,
        "reference_sha256": sha256(reference), "driver_sha256": sha256(driver),
        "reference": picture, "driver_video": driver_meta,
        "entry": str(entry), "config": str(cfg), "runtime": str(runtime),
        "output": str(output), "width": width, "height": height,
        "frames": frames, "fps": 24, "steps": 10, "guidance": 1.,
        "seed": seed, "prompt": prompt.strip(),
        "hardware_notice": "Upstream 480p tested with 2x A800; different hardware is unverified",
        "gpu_executed": False, "human_approved": False, "published": False,
    }


def hardware_preflight() -> dict:
    if not shutil.which("nvidia-smi"):
        raise NeuralBlocked("Real NVIDIA GPU nodes not found")
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"
    ], check=True, capture_output=True, text=True, timeout=20)
    memory = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    if len(memory) < 2 or min(memory[:2]) < 70 * 1024:
        raise NeuralBlocked("This upstream config needs 2 reviewed high-memory GPUs; not a single Kaggle T4")
    return {"gpus": len(memory), "min_free_mib_first_two": min(memory[:2])}


def generate(*, runtime: Path, reference: Path, driver: Path, output: Path,
             prompt: str, seed: int = 42, width: int = 480, height: int = 848,
             frames: int = 81, execute: bool = False, accepted_license: bool = False,
             timeout: int = 7200) -> dict:
    spec = plan(runtime=runtime, reference=reference, driver=driver, output=output,
                prompt=prompt, seed=seed, width=width, height=height, frames=frames)
    if not accepted_license:
        raise NeuralBlocked("Read and accept the installed Wan-Animate-2 weights licence first")
    if not execute:
        return spec
    if type(timeout) is not int or not 120 <= timeout <= 14400:
        raise ValueError("Bound the model runtime to 120–14400 seconds")
    device = hardware_preflight()
    runtime = Path(runtime).resolve()
    cmd = subprocess.run(["git", "-C", str(runtime), "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True, timeout=30)
    revision = cmd.stdout.strip()
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise NeuralBlocked("Use a pinned Git checkout of the official model runtime")
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tw-wan-animate2-", dir=output.parent) as folder:
        work = Path(folder)
        argv = [
            sys.executable, spec["entry"], "--config", spec["config"],
            "--refer-img-file", str(Path(reference).resolve()),
            "--refer-video-file", str(Path(driver).resolve()),
            "--prompt", spec["prompt"], "--width", str(width), "--height", str(height),
            "--fps", "24", "--clip_len", str(frames),
            "--sample_guide_scale", "1.0", "--step", "10",
            "--seed", str(seed), "--output-dir", str(work),
        ]
        with (work / "upstream.log").open("wb") as logfile:
            result = subprocess.run(argv, cwd=runtime / "infer",
                                    stdout=logfile, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise NeuralBlocked("Wan-Animate-2 upstream inference failed; inspect local worker logs")
        candidates = [p for p in work.rglob("*.mp4") if p.is_file()]
        if len(candidates) != 1:
            raise NeuralBlocked("Wan-Animate-2 must generate one unambiguous output MP4")
        clip = candidates[0]
        stats = _probe(clip)
        if (stats["fps"] != "24" or stats["frames"] != frames
                or abs(stats["width"] / stats["height"] - width / height) > .025):
            raise NeuralBlocked("Official Wan-Animate-2 output has unexpected frame clock")
        # Preserve exactly the native video, never interpolate to fake performance.
        os.link(clip, output)
        proof = {"schema": 1, "engine": "neural_animate2", "model": MODEL,
                 "source_revision": revision, "config_sha256": sha256(Path(spec["config"])),
                 "reference_sha256": spec["reference_sha256"],
                 "driver_sha256": spec["driver_sha256"], "clip_sha256": sha256(output),
                 "video": stats, "seed": seed, "steps": 10, "guidance": 1.,
                 "inference_performed": True, "weights_verified": False,
                 "human_approved": False, "published": False, "hardware": device}
        receipt = output.with_suffix(".receipt.json")
        try:
            with receipt.open("x", encoding="utf-8") as fd:
                json.dump(proof, fd, indent=2)
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    return {"status": "GENERATED_REVIEW_REQUIRED", "receipt": str(receipt), **proof}


def import_take(clip: Path, receipt: Path, reference: Path, driver: Path, *, duration: float) -> dict:
    if receipt.stat().st_size > 64 * 1024:
        raise ValueError("Wan-Animate-2 receipt too large")
    data = json.loads(receipt.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("engine") != "neural_animate2"
            or data.get("schema") != 1 or data.get("model") != MODEL
            or data.get("inference_performed") is not True
            or data.get("reference_sha256") != sha256(reference)
            or data.get("driver_sha256") != sha256(driver)
            or data.get("clip_sha256") != sha256(clip)
            or not re.fullmatch(r"[a-f0-9]{40}", str(data.get("source_revision")))
            or not re.fullmatch(r"[a-f0-9]{64}", str(data.get("config_sha256")))
            or data.get("video") != _probe(clip)
            or data["video"]["seconds"] + .001 < duration):
        raise ValueError("Wan-Animate-2 source and full clip receipts disagree")
    return {"model": MODEL, "clip_sha256": data["clip_sha256"],
            "native_video": data["video"], "character_review": "required"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Private official Wan-Animate-2 distilled runner")
    for name in ("runtime", "reference", "driver", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=848)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--accept-model-license", action="store_true")
    p.add_argument("--execute-local-gpu", action="store_true")
    args = p.parse_args(argv)
    try:
        result = generate(runtime=args.runtime, reference=args.reference, driver=args.driver,
                          output=args.output, prompt=args.prompt, seed=args.seed,
                          width=args.width, height=args.height, frames=args.frames,
                          timeout=args.timeout, execute=args.execute_local_gpu,
                          accepted_license=args.accept_model_license)
    except (ValueError, NeuralBlocked, OSError, subprocess.SubprocessError) as exc:
        result = {"status": "BLOCKED", "reason": str(exc), "gpu_executed": False, "published": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
