"""Run official character-video inference locally and lock takes to the edit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageOps

from studio.audio_master import describe_audio
from studio.production_queue import private_path
from studio.release_gate import sha256

ECHO = "BadToBest/EchoMimicV3/echomimicv3-flash-pro"
WAN = "Wan-AI/Wan2.2-Animate-14B"
MIN_FREE_MIB = {ECHO: 12 * 1024, WAN: 24 * 1024}
ECHO_CHECKPOINT_SHA = "5ebdbb2fc709108bf2a1728fd92eb2874804e4bc0324e92a2cd55425968c85a4"
WAN_PREPROCESSOR = "wan/modules/animate/preprocess/preprocess_data.py"
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1"}
CLIP_LIMIT_SECONDS = 60
FRAME_SIZE = (432, 768)


def _run(args, *, cwd=None, timeout=120):
    result = subprocess.run(args, cwd=cwd, env={**os.environ, **OFFLINE},
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} exited {result.returncode}: {result.stderr[-1500:]}")
    return result


def _native_video(path: Path, duration: float, *, portrait: bool) -> dict:
    result = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                   "-show_entries", "stream=width,height,avg_frame_rate,start_time,duration,nb_read_frames",
                   "-of", "json", str(path)])
    try:
        stream, = json.loads(result.stdout)["streams"]
        fps = Fraction(stream["avg_frame_rate"])
        frames = int(stream["nb_read_frames"])
        start = float(stream["start_time"])
        length = float(stream["duration"])
        w, h = int(stream["width"]), int(stream["height"])
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        raise ValueError("Missing native video frame clock") from exc
    if (result.stderr or not 12 <= fps <= 60 or frames < 2 or not 256 <= min(w, h) <= max(w, h) <= 3840
            or not abs(start) <= .04 or not duration - .001 <= length <= CLIP_LIMIT_SECONDS
            or abs(float(frames / fps) - length) > 1 / fps
            or portrait and abs(w / h - 9 / 16) > .08):
        raise ValueError("Take needs a decoded, full-length video with the expected framing")
    return {"frames": frames, "fps": str(fps), "size": [w, h], "duration_s": round(length, 6)}


def _echo_frames(duration):
    # The upstream VAE accepts 4n + 1 frames and caps them by the WAV duration.
    return 1 + 4 * math.ceil((math.ceil(25 * duration) - 1) / 4)


def _decode_take(path, model, duration):
    video = _native_video(path, duration, portrait=True)
    if model == ECHO and (video["fps"] != "25" or video["frames"] != _echo_frames(duration)):
        raise ValueError("EchoMimic output differs from the requested native 25 FPS frame clock")
    return video


def _source_inputs(model, reference, driver, duration, offset_ms, seed):
    if (not isinstance(model, str) or model not in MIN_FREE_MIB or type(duration) not in (int, float)
            or not math.isfinite(duration) or not .5 <= duration <= 10):
        raise ValueError("Choose a supported model and a 0.5–10 second shot")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be a nonnegative 32-bit integer")
    if type(offset_ms) is not int or not 0 <= offset_ms < duration * 1000:
        raise ValueError("Voice offset must be inside the shot")
    if reference.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not reference.is_file():
        raise ValueError("A local full-frame character reference is required")
    if model == ECHO:
        if driver.suffix.lower() != ".wav" or not driver.is_file():
            raise ValueError("EchoMimic needs the locked dialogue WAV")
        if describe_audio(driver)["duration"] + offset_ms / 1000 > duration + .003:
            raise ValueError("Extend the shot to fit the original voice and offset")
    else:
        if offset_ms or driver.suffix.lower() != ".mp4" or not driver.is_file():
            raise ValueError("Wan Animate needs a local driving MP4 with zero voice offset")
        _native_video(driver, duration, portrait=False)
    return {"model": model, "reference_sha256": sha256(reference), "driver_sha256": sha256(driver),
            "voice_offset_ms": offset_ms, "duration_ms": round(duration * 1000), "seed": seed}


def _cuda_dtype(python, model):
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("Local NVIDIA CUDA GPU is unavailable")
    command = ("import json, torch; "
               "assert torch.cuda.is_available(), 'CUDA unavailable in the selected environment'; "
               "free, _ = torch.cuda.mem_get_info(); "
               "bf16 = torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported(); "
               "print(json.dumps([free // (1024**2), bf16]))")
    free_mib, has_bf16 = json.loads(_run([str(python), "-c", command]).stdout)
    if free_mib < MIN_FREE_MIB[model]:
        raise RuntimeError(f"Local model needs at least {MIN_FREE_MIB[model]} MiB of free VRAM for preflight")
    if model == WAN and not has_bf16:
        raise RuntimeError("The Wan runtime needs native bfloat16 support")
    return "bfloat16" if has_bf16 else "float16"


def _runtime_info(runtime, entry):
    if not (runtime / entry).is_file():
        raise ValueError(f"Local upstream runtime is missing {entry}")
    revision = _run(["git", "rev-parse", "HEAD"], cwd=runtime).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a Git checkout of the official runtime")
    return {"revision": revision, "entry_sha256": sha256(runtime / entry)}


def _prepare_reference(source, target):
    with Image.open(source) as image:
        oriented = ImageOps.exif_transpose(image)
        if (min(oriented.size) < 256 or abs(oriented.width / oriented.height - 9 / 16) > .025
                or oriented.convert("RGBA").getchannel("A").getextrema()[0] < 255):
            raise ValueError("Supply the composed 9:16 picture with an opaque background")
        oriented.convert("RGB").resize(FRAME_SIZE, Image.Resampling.LANCZOS).save(target)


def _render_echo(python, runtime, weights, reference, driver, output, prompt, request, dtype):
    base = weights / "Wan2.1-Fun-V1.1-1.3B-InP"
    encoder = weights / "chinese-wav2vec2-base"
    checkpoint = weights / "transformer/diffusion_pytorch_model.safetensors"
    config = runtime / "config/config.yaml"
    if (not checkpoint.is_file() or not base.is_dir() or not encoder.is_dir() or not config.is_file()):
        raise ValueError("Incomplete local EchoMimicV3-Flash bundle")
    if sha256(checkpoint) != ECHO_CHECKPOINT_SHA:
        raise ValueError("EchoMimic Flash checkpoint SHA-256 differs from the official file")
    frames = _echo_frames(request["duration_ms"] / 1000)
    conditioning = output / "conditioning.wav"
    # Only leading/trailing silence and a rate conversion for the speech encoder.
    _run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(driver), "-ac", "1",
          "-af", f"aresample=16000,adelay={request['voice_offset_ms']}:all=1,apad,"
                 f"atrim=end_sample={(frames + 1) * 640}",
          "-c:a", "pcm_s16le", str(conditioning)])
    _run([str(python), str(runtime / "infer_flash.py"), "--config_path", str(config),
          "--model_name", str(base), "--wav2vec_model_dir", str(encoder),
          "--transformer_path", str(checkpoint), "--image_path", str(reference),
          "--audio_path", str(conditioning), f"--prompt={prompt}",
          "--save_path", str(output), "--video_length", str(frames), "--fps", "25",
          "--num_inference_steps", "8", "--sample_size", "768", "432",
          "--weight_dtype", dtype, "--seed", str(request["seed"])], cwd=runtime, timeout=3600)
    return output / f"{reference.stem}_output.mp4", {"transformer_sha256": ECHO_CHECKPOINT_SHA,
                                                     "config_sha256": sha256(config)}


def _render_wan(python, runtime, weights, reference, driver, output, request):
    preprocessor = runtime / WAN_PREPROCESSOR
    process_weights = weights / "process_checkpoint"
    config, index = (weights / name for name in ("config.json", "diffusion_pytorch_model.safetensors.index.json"))
    if not all(path.is_file() for path in (
            preprocessor, config, index, process_weights / "pose2d/vitpose_h_wholebody.onnx",
            process_weights / "det/yolov10m.onnx")):
        raise ValueError("Incomplete local Wan2.2 Animate runtime or weights")
    driving = output / "performance.mp4"
    _run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(driver), "-an",
          "-t", f"{request['duration_ms'] / 1000:.3f}", "-vf", "fps=30",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", str(driving)])
    processed = output / "prepared"
    _run([str(python), str(preprocessor), "--ckpt_path", str(process_weights),
          "--video_path", str(driving), "--refer_path", str(reference), "--save_path", str(processed),
          "--resolution_area", "480", "832", "--fps", "30"], cwd=runtime, timeout=900)
    if not all((processed / name).is_file() for name in ("src_face.mp4", "src_pose.mp4")):
        raise RuntimeError("Wan preprocessing did not yield both face and pose tracks")
    clip = output / "generated.mp4"
    _run([str(python), str(runtime / "generate.py"), "--task", "animate-14B",
          "--ckpt_dir", str(weights), "--src_root_path", str(processed), "--refert_num", "1",
          "--offload_model", "True", "--t5_cpu", "--convert_model_dtype",
          "--base_seed", str(request["seed"]), "--save_file", str(clip)],
         cwd=runtime, timeout=3600)
    return clip, {"config_sha256": sha256(config), "shard_index_sha256": sha256(index),
                  "preprocess_sha256": sha256(preprocessor)}


def generate_take(model: str, *, reference: Path, driver: Path, duration: float, voice_offset_ms: int,
                  seed: int, prompt: str | None, runtime: Path, weights: Path, python: Path, output: Path) -> dict:
    reference, driver = reference.resolve(), driver.resolve()
    runtime, weights = runtime.resolve(), weights.resolve()
    # Do not resolve a dangling final symlink as a new destination.
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new output directory or keep the completed take")
    output = private_path(output)
    request = _source_inputs(model, reference, driver, duration, voice_offset_ms, seed)
    if model == ECHO and (not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 1200):
        raise ValueError("EchoMimic needs a bounded shot direction")
    if model == WAN and prompt is not None:
        raise ValueError("Wan Animate follows the driving take and does not consume a text prompt")
    # Keep a virtual environment's Python path, including its pyvenv.cfg location.
    python = python.absolute()
    dtype = _cuda_dtype(python, model)
    upstream = _runtime_info(runtime, "infer_flash.py" if model == ECHO else "generate.py")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=".tw-character-", dir=output.parent) as temporary:
        root = Path(temporary)
        plate = root / "reference.png"
        _prepare_reference(reference, plate)
        if model == ECHO:
            clip, checkpoints = _render_echo(python, runtime, weights, plate, driver, root, prompt, request, dtype)
        else:
            clip, checkpoints = _render_wan(python, runtime, weights, plate, driver, root, request)
        bundle = root / "take"
        bundle.mkdir()
        silent = bundle / "take.mp4"
        _run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(clip),
              "-map", "0:v:0", "-c:v", "copy", "-an", "-movflags", "+faststart", str(silent)])
        decoded = _decode_take(silent, model, request["duration_ms"] / 1000)
        if (request["reference_sha256"], request["driver_sha256"]) != (sha256(reference), sha256(driver)):
            raise ValueError("Character source changed during inference")
        proof = {"schema": 1, **request, "clip_sha256": sha256(silent), "native_video": decoded,
                 "upstream": upstream, "checkpoints": checkpoints, "dtype": dtype,
                 "runtime_seconds": round(time.monotonic() - started, 2),
                 "quality_approved": False, "published": False}
        if prompt is not None:
            proof["direction_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        (bundle / "take.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
        bundle.rename(output)
    return {"status": "review_required", **proof}


def import_take(clip: Path, receipt: Path, reference: Path, driver: Path, *, duration: float,
                voice: Path | None, voice_offset: float, voice_bus: str, regions: list[dict]) -> dict:
    if receipt.stat().st_size > 64 * 1024:
        raise ValueError("Character take receipt is too large")
    data = json.loads(receipt.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or type(data.get("schema")) is not int or data["schema"] != 1
            or not isinstance(data.get("model"), str) or data["model"] not in MIN_FREE_MIB):
        raise ValueError("Unknown character take receipt")
    model = data["model"]
    echo = model == ECHO
    if (echo and (voice is None or driver != voice or voice_bus != "dialogue")
            or not echo and voice is not None and voice_bus != "narration"):
        raise ValueError("Character take does not match the shot dialogue")
    expected = _source_inputs(model, reference, driver, duration,
                              round(voice_offset * 1000) if echo else 0, data.get("seed"))
    if (any(data.get(k) != v for k, v in expected.items()) or data.get("clip_sha256") != sha256(clip)
            or data.get("native_video") != _decode_take(clip, model, expected["duration_ms"] / 1000)):
        raise ValueError("Character take is stale or does not match the clip, reference and driving input")
    upstream, checkpoints = data.get("upstream"), data.get("checkpoints")
    if (not isinstance(upstream, dict) or not isinstance(checkpoints, dict)
            or not re.fullmatch(r"[0-9a-f]{40}", str(upstream.get("revision")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(upstream.get("entry_sha256")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(checkpoints.get("config_sha256")))
            or echo and (checkpoints.get("transformer_sha256") != ECHO_CHECKPOINT_SHA
                         or not re.fullmatch(r"[0-9a-f]{64}", str(data.get("direction_sha256"))))
            or not echo and any(not re.fullmatch(r"[0-9a-f]{64}", str(checkpoints.get(key)))
                                for key in ("shard_index_sha256", "preprocess_sha256"))):
        raise ValueError("Character take needs its upstream and checkpoint fingerprints")
    validate_performance_regions(regions, echo=echo, reference=reference)
    return {"model": model, "clip_sha256": data["clip_sha256"], "native_video": data["native_video"],
            "upstream_revision": upstream["revision"], "character_review": "required"}


def validate_performance_regions(regions, *, echo, reference):
    groups = {region["kind"]: region for region in regions}
    camera = groups.get("camera")
    face, wardrobe = (groups.get(kind) for kind in ("face", "wardrobe"))
    digest = sha256(reference)
    if (not all((camera, face, wardrobe)) or not face.get("motion_windows") or not wardrobe.get("motion_windows")
            or wardrobe.get("relative_to") not in (None, camera["id"])
            or any(r.get("identity_reference", {}).get("sha256") != digest for r in (face, wardrobe))):
        raise ValueError("Character take needs camera, head and wardrobe checks pinned to its reference")
    if not echo:
        if (face.get("relative_to") != wardrobe["id"] or face["motion_windows"] != wardrobe["motion_windows"]):
            raise ValueError("Reaction must measure head action independently of shoulders in the same windows")
        return
    mouth, jaw, cheek = (groups.get(kind) for kind in ("mouth", "jaw", "cheek"))
    if (not all((mouth, jaw, cheek)) or not mouth.get("motion_windows")
            or any(r.get("relative_to") != face["id"] or r.get("motion_windows") != mouth["motion_windows"]
                   for r in (mouth, jaw, cheek))):
        raise ValueError("Dialogue needs separate mouth, jaw and cheek checks relative to the upper face")
    mx0, my0, mx1, my1 = mouth["box"]
    for region in (jaw, cheek):
        x0, y0, x1, y1 = region["box"]
        if min(mx1, x1) > max(mx0, x0) and min(my1, y1) > max(my0, y0):
            raise ValueError("Jaw and cheek evidence must be outside the lip region")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("echo-flash", "wan-animate"), required=True)
    for name in ("reference", "driver", "runtime", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--voice-offset-ms", type=int, default=0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--execute-local-gpu", action="store_true")
    args = parser.parse_args()
    model = ECHO if args.model == "echo-flash" else WAN
    try:
        if not args.execute_local_gpu:
            result = {"status": "planned", "model": model, "minimum_free_vram_mib": MIN_FREE_MIB[model],
                      "local_nvidia_detected": bool(shutil.which("nvidia-smi")), "model_executed": False}
        else:
            result = generate_take(
                model, reference=args.reference, driver=args.driver, duration=args.duration,
                voice_offset_ms=args.voice_offset_ms, seed=args.seed,
                prompt=args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else None, runtime=args.runtime,
                weights=args.weights, python=args.python, output=args.output,
            )
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        result = {"status": "blocked", "model": model, "reason": str(exc), "published": False}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
