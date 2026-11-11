"""Private Kaggle GPU worker for ELIZABETH BONES through native TWO WORLDS.

Real local SDXL-generated artwork, real local Qwen3-TTS Russian cast,
TWO WORLDS' own hybrid_master renderer and own QA receipts. This worker NEVER
uses InVideo/Runway editors, does not fake model success, and does not publish.
"""
from __future__ import annotations

import base64
import gc
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

CONFIG_B64 = "__ELIZABETH_CONFIG_B64__"
SOURCE_B64 = "__ELIZABETH_SOURCE_B64__"
ROOT = Path("/kaggle/temp/elizabeth_native_studio")
WORK = Path("/kaggle/working")
RECEIPT_NAME = "ELIZABETH_BONES_STUDIO_JOB.json"
PACKAGE_NAME = "ELIZABETH_BONES_STUDIO_PILOT.zip"
PREVIEW = "ELIZABETH_BONES_SOBESEDOVANIE_STUDIO_PREVIEW.mp4"


def checked(*args, timeout=1500):
    subprocess.run(list(args), check=True, timeout=timeout)


def stage_sources() -> None:
    from pathlib import PurePosixPath
    ROOT.mkdir(parents=True, exist_ok=False)
    data = base64.b64decode(SOURCE_B64, validate=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or ".." in path.parts
                    or path.parts[0] != "studio" or not path.name.endswith(".py")
                    or info.file_size > 180000 or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe native Studio worker source")
        z.extractall(ROOT)
    sys.path.insert(0, str(ROOT))


def setup_dependencies() -> None:
    # Use Kaggle's preinstalled CUDA torch, do not reinstall torch or download
    # unnecessary ASR/video models. Only the reviewed Studio providers below.
    checked(sys.executable, "-m", "pip", "install", "-q",
            "soundfile>=0.12", "pillow>=10,<12", "numpy<2.1",
            "accelerate>=1.0,<2", "diffusers>=0.35,<0.38",
            "transformers>=4.55,<5", "safetensors>=0.4",
            "qwen-tts", timeout=1500)


def voices(work: Path) -> dict[str, float]:
    from studio.elizabeth_episode_pilot import SCENES
    from studio import russian_qwen_tts as tts
    import torch
    import soundfile as sf
    destination = work / "voices"
    destination.mkdir()
    results = {}
    for scene_id, speaker, words, *_ in SCENES:
        if not speaker:
            continue
        person = "Serena" if speaker == "ЭЛИЗАБЕТ" else "Ryan"
        style = "warm" if speaker == "ЭЛИЗАБЕТ" else "joyful"
        wav = destination / (scene_id + ".wav")
        report = tts.synthesize(words, person, wav, style=style)
        with sf.SoundFile(wav) as sound:
            actual = sound.frames / sound.samplerate
            if sound.channels != 1 or sound.frames < 3000 or abs(actual - report["duration_seconds"]) > .006:
                raise RuntimeError("Qwen voice failed genuine WAV and clock verification")
        results[scene_id] = actual
        print("ELIZABETH_REAL_QWEN_RUSSIAN", scene_id, speaker, round(actual, 3), flush=True)
    tts._get_model.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()
    if len(results) != 6:
        raise RuntimeError("Native Studio did not create exactly six real Russian voice files")
    return results


def render_art(work: Path):
    import torch
    from diffusers import StableDiffusionXLPipeline
    from PIL import Image, ImageChops, ImageStat
    from studio.elizabeth_episode_pilot import (
        IMAGE_MODEL, NEGATIVE, SHOT_PROMPTS, SCENES, WIDTH, HEIGHT
    )
    art = work / "art"
    art.mkdir()
    print("TW_STUDIO_SDXL_LOAD", IMAGE_MODEL, flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        IMAGE_MODEL, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    figures = []
    for i, scene in enumerate(SCENES):
        scene_id = scene[0]
        seed = 441101 + i
        image = pipe(
            prompt=SHOT_PROMPTS[i], negative_prompt=NEGATIVE,
            width=WIDTH, height=HEIGHT, num_inference_steps=25,
            guidance_scale=6.0,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        if not isinstance(image, Image.Image) or image.size != (WIDTH, HEIGHT):
            raise RuntimeError("SDXL returned no real 9:16 vertical photograph")
        output = art / (scene_id + ".png")
        image.save(output)
        if output.stat().st_size < 50_000 or ImageStat.Stat(image.convert("RGB")).stddev[0] < 8:
            raise RuntimeError("SDXL shot is too small or uniform")
        figures.append(image.resize((72, 128)).convert("RGB"))
        print("ELIZABETH_REAL_SDXL_STILL", scene_id, output.stat().st_size, flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    # This is not an identity test; it detects catastrophic duplicate outputs.
    for i, a in enumerate(figures):
        for j, b in enumerate(figures[:i]):
            delta = sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3
            if delta < 2.5:
                raise RuntimeError("Two independent shots are effectively identical: " +
                                   SCENES[i][0] + "/" + SCENES[j][0])
    return {scene[0]: (441101 + i) for i, scene in enumerate(SCENES)}


def render_cover(work: Path):
    from PIL import Image, ImageDraw, ImageFont
    original = work / "art" / "S00.png"
    frame = Image.open(original).convert("RGBA")
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    width = frame.width
    d.rounded_rectangle((26, 65, width - 26, 242), radius=19,
                        fill=(22, 10, 18, 209), outline=(215, 175, 116, 190), width=3)
    font_path = next((x for x in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
    ) if Path(x).exists()), None)
    if font_path is None:
        raise RuntimeError("No Cyrillic font installed for Studio title overlay")
    for y, text, size in (
        (83, "ELIZABETH BONES", 30),
        (137, "СОБЕСЕДОВАНИЕ", 37),
        (199, "СЕРИЯ 01 · ЭЛИЗАБЕТ И КОСТЯ", 17),
    ):
        font = ImageFont.truetype(font_path, size)
        d.text((width / 2, y), text, font=font, fill=(248, 224, 182, 255),
               anchor="mt", stroke_width=1, stroke_fill=(23, 10, 15, 255))
    merged = Image.alpha_composite(frame, layer).convert("RGB")
    merged.save(original)


def create_original_score(work: Path, seconds: float):
    import numpy as np
    import soundfile as sf
    sample_rate = 24000
    n = math.ceil(seconds * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    # Original non-copyrighted score: chamber-like minor ostinato, slow
    # dynamic harmonics and room ambience, synthesized in OUR Studio code.
    score = np.zeros(n, dtype=np.float64)
    for tone, level in ((110.0, .16), (164.81, .11), (220.0, .07),
                        (329.63, .024), (440.0, .012)):
        score += level * np.sin(2 * np.pi * tone * t +
                              0.025 * np.sin(2 * np.pi * .17 * t))
    edge = np.minimum(1.0, np.minimum(t / .85, (seconds - t) / 1.25))
    edge = np.maximum(0.0, edge)
    pulse = (.70 + .25 * np.sin(2 * np.pi * .28 * t) ** 2)
    score *= edge * pulse
    # Tiny stereo differences broaden the acoustic space without phase cancellation.
    left = np.clip(score, -.8, .8)
    right = np.clip(np.roll(score, 131) * .96, -.8, .8)
    sf.write(work / "score.wav", np.column_stack((left, right)).astype("float32"),
             sample_rate, subtype="PCM_16")


def delivery_decode(path: Path):
    out = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate",
        "-of", "json", str(path)
    ], check=True, capture_output=True, text=True, timeout=100)
    streams = json.loads(out.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("Native Studio MP4 contains no single valid video stream")
    row = streams[0]
    from studio.elizabeth_episode_pilot import WIDTH, HEIGHT, FPS
    if (row.get("width") != WIDTH or row.get("height") != HEIGHT
            or row.get("avg_frame_rate") != str(FPS) + "/1"
            or int(row.get("nb_read_frames", 0)) < 350):
        raise RuntimeError("Native Studio MP4 has invalid frame clock or dimensions")
    checked("ffmpeg", "-nostdin", "-loglevel", "error", "-xerror",
            "-i", str(path), "-f", "null", "-", timeout=300)
    return int(row["nb_read_frames"])


def production(config):
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 14 * 2**30:
        raise RuntimeError("Private Kaggle 16 GB-class T4 GPU required")
    stage_sources()
    setup_dependencies()
    from studio.elizabeth_episode_pilot import SCENES, PACK, RECEIPT, native_render
    working = WORK / "elizabeth_studio_private"
    working.mkdir(exist_ok=False)
    clock = voices(working)
    seeds = render_art(working)
    render_cover(working)
    preview_seconds = 2.2 + sum(
        math.ceil(max(minimum, clock[scene_id] + .24) * 24) / 24
        for scene_id, speaker, text, shot, cam, minimum in SCENES if speaker
    )
    create_original_score(working, preview_seconds + 1.0)
    final_receipt = native_render(working, clock)
    film = working / PREVIEW
    frames = delivery_decode(film)
    if (frames != round(final_receipt["duration_seconds"] * 24)
            or abs(final_receipt["duration_seconds"] - preview_seconds) > .2):
        raise RuntimeError("Native Studio duration does not match independently measured voices")
    proof = {
        "schema": 1, "run_nonce": config["run_nonce"],
        "studio_renderer": "studio.hybrid_master",
        "pipeline": "studio.hybrid_master.produce",
        "external_editor": False, "native_studio": True,
        "image_model": config["image_model"], "voice_model": config["voice_model"],
        "source_revision": config["revision"],
        "shot_ids": [v[0] for v in SCENES], "seed_by_shot": seeds,
        "russian_dialogue": {v[0]: v[2] for v in SCENES if v[1]},
        "voice_clock": {k: round(v, 3) for k, v in clock.items()},
        "human_approved": False, "editorial_approved": False,
        "identity_verified": False, "lip_sync_used": False,
        "animation_method": "studio.smooth_camera.ken_burns",
        "published": False
    }
    (working / "production_provenance.json").write_text(
        json.dumps(proof, indent=2, ensure_ascii=False) + "\n")
    expected = [film, film.with_suffix(".srt"), film.with_suffix(".json"),
                working / "studio_shot_manifest.json",
                working / "production_provenance.json"]
    archive = WORK / PACK
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in expected:
            if not path.is_file():
                raise FileNotFoundError("Studio render incomplete: " + path.name)
            z.write(path, arcname=path.name)
    result = {
        "schema": 1, "run_nonce": config["run_nonce"], "success": True,
        "provider": "native_two_worlds_studio",
        "pipeline": "studio.hybrid_master.produce",
        "gpu_name": torch.cuda.get_device_name(0),
        "shots": 7, "voices": 6, "native_frames": frames,
        "duration": final_receipt["duration_seconds"],
        "video_sha256": hashlib.sha256(film.read_bytes()).hexdigest(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "published": False, "editorial_approved": False,
    }
    (WORK / RECEIPT).write_text(json.dumps(result, indent=2) + "\n")
    print("TW_ELIZABETH_REAL_NATIVE_STUDIO_RENDER_REVIEW_REQUIRED", result["duration"], flush=True)


def main():
    config = json.loads(base64.b64decode(CONFIG_B64, validate=True))
    try:
        if (config.get("schema") != 1 or not config.get("lock_publishing")
                or [config.get("width"), config.get("height"), config.get("fps")] != [576, 1024, 24]
                or len(config.get("scenes", [])) != 7):
            raise ValueError("Unreviewed private native Studio job")
        production(config)
    except BaseException as exc:
        (WORK / RECEIPT_NAME).write_text(json.dumps({
            "schema": 1, "run_nonce": config.get("run_nonce"),
            "success": False, "published": False,
            "error_type": type(exc).__name__, "error_summary": str(exc)[:300]
        }, indent=2) + "\n")
        print("TW_ELIZABETH_STUDIO_EXECUTION_FAILED", type(exc).__name__, str(exc)[:200], flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
