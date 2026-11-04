"""Self-hosted, zero-API media backends for Two Worlds short stories.

Model weights must already be present on the user's own disk. The module
never downloads weights, invokes a paid endpoint, or invents audio/images.
It intentionally does NOT claim character continuity, artistic quality, or
publish approval: these remain review gates.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

MODEL_NAMES = {"image": "black-forest-labs/FLUX.2-klein-4B",
               "voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"}
MODEL_LICENSES = {"image": "Apache-2.0", "voice": "Apache-2.0"}
VOICE_STYLES = {
    "neutral": "Speak in natural Russian with clear conversational phrasing.",
    "tender": "Speak softly, warmly and sincerely, with restrained human emotion.",
    "sad": "Speak in natural Russian with understated grief and gentle pauses, not melodrama.",
    "mysterious": "Speak with quiet intrigue and thoughtful pauses, like an intimate thriller.",
    "tense": "Speak with subtle urgency and controlled tension, without shouting.",
    "hopeful": "Speak with restrained hope, openness and warmth.",
}
SPEAKERS = {"Serena", "Ryan", "Vivian", "Aiden", "Sohee", "Ono_Anna", "Dylan", "Eric", "Uncle_Fu"}


class OfflineBlocked(RuntimeError):
    """A truthful blocking reason, not a reason to synthesize placeholder media."""


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def local_model(root: Path, kind: str) -> Path:
    if kind not in MODEL_NAMES:
        raise ValueError("Unsupported model kind")
    p = Path(root).expanduser().resolve() / kind
    if not p.is_dir() or not (p / "config.json").is_file():
        raise OfflineBlocked(f"OFFLINE_MODEL_MISSING: {p}; stage a legal local {MODEL_NAMES[kind]} checkpoint")
    return p


def accelerator_status() -> dict:
    """Diagnostic only; imports torch if present, never downloads models."""
    status = {"cuda": False, "gpu": None, "vram_gb": 0.0,
              "image_local_recommended_vram_gb": 12.0,
              "voice_local_recommended_vram_gb": 6.0,
              "high_quality_video_model_local_recommended_vram_gb": 24.0}
    try:
        import torch
        status["cuda"] = bool(torch.cuda.is_available())
        if status["cuda"]:
            status["gpu"] = torch.cuda.get_device_name(0)
            status["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except ImportError:
        pass
    return status


def split_narration(text: str, max_chars: int = 230) -> list[str]:
    """Split into phrase-sized cues, preserving all original words in order."""
    if not isinstance(text, str) or not text.strip() or max_chars < 50:
        raise ValueError("Missing text or invalid phrase limit")
    raw = re.sub(r"\s+", " ", text.strip())
    if not re.search(r"[А-Яа-яЁё]", raw):
        raise ValueError("Russian narration requires Russian Cyrillic source")
    pieces = re.split(r"(?<=[.!?…])\s+", raw)
    out, cur = [], ""
    for piece in pieces:
        for word in piece.split(" "):
            if len(word) > max_chars:
                raise ValueError("Very long unbreakable word in source")
            next_text = (cur + " " + word).strip()
            if len(next_text) > max_chars and cur:
                out.append(cur)
                cur = word
            else:
                cur = next_text
    if cur:
        out.append(cur)
    if not out or len(out) > 250:
        raise ValueError("Too many narrative segments")
    assert " ".join(out) == raw
    return out


def _load_qwen(model_dir: Path, device: str):
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise OfflineBlocked("Qwen3 TTS dependencies missing: pip install qwen-tts soundfile") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise OfflineBlocked("CUDA unavailable: cannot run Qwen3 voice on requested GPU")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == "cuda" else torch.float32
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return Qwen3TTSModel.from_pretrained(str(model_dir),
                                       device_map="cuda:0" if device == "cuda" else "cpu",
                                       dtype=dtype)


@lru_cache(maxsize=1)
def qwen_model(model: str, device: str):
    return _load_qwen(Path(model), device)


def real_voice(text: str, dest: Path, model_dir: Path, *, speaker: str = "Serena",
               style: str = "neutral", device: str = "cuda", model=None) -> dict:
    """Render local Russian voice with *measured chunk timings*, not word guesses.

    The cue text matches the submitted passage. Time positions are exact WAV
    chunk boundaries and do not purport to be phoneme/word forced alignment.
    """
    if speaker not in SPEAKERS or style not in VOICE_STYLES:
        raise ValueError("Invalid Qwen speaker or Russian emotional preset")
    chunks = split_narration(text)
    if model is None:
        model = qwen_model(str(local_model(model_dir, "voice")), device)
    try:
        import numpy as np
    except ImportError as exc:
        raise OfflineBlocked("numpy is required for voice synthesis") from exc
    samples = []
    rate = None
    timeline = []
    cursor = 0.0
    pause = 0.14
    for i, chunk in enumerate(chunks):
        # The real API returns a list of wave arrays plus a sample rate.
        waves, sr = model.generate_custom_voice(text=chunk, language="Russian",
                                                speaker=speaker,
                                                instruct=VOICE_STYLES[style])
        sr = int(sr)
        if not 16000 <= sr <= 96000 or not waves:
            raise OfflineBlocked("Qwen produced invalid sample rate or no speech")
        wav = np.asarray(waves[0], dtype=np.float32).reshape(-1)
        if not (len(wav) >= int(sr * .08) and np.isfinite(wav).all()):
            raise OfflineBlocked("Qwen produced broken speech audio")
        if np.max(np.abs(wav)) < .0005:
            raise OfflineBlocked("Qwen produced nearly silent speech")
        if rate is None:
            rate = sr
        if sr != rate:
            raise OfflineBlocked("Sample rate changed between voice segments")
        if i:
            blank = np.zeros(round(pause * rate), dtype=np.float32)
            samples.append(blank)
            cursor += len(blank) / rate
        start = cursor
        samples.append(wav)
        cursor += len(wav) / rate
        timeline.append({"start": round(start, 4), "end": round(cursor, 4), "text": chunk})
    output = np.concatenate(samples)
    if cursor >= 600:
        raise OfflineBlocked("Scene voice exceeds 10 minutes")
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".voice-", suffix=".wav", dir=dest.parent,
                                     delete=False) as temp:
        tmp = Path(temp.name)
    try:
        pcm = (output.clip(-1, 1) * 32767).astype("<i2").tobytes()
        with wave.open(str(tmp), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes(pcm)
        with wave.open(str(tmp), "rb") as check:
            if (check.getnframes() != len(output) or check.getframerate() != rate
                    or check.getnchannels() != 1):
                raise OfflineBlocked("Voice WAV integrity check failed")
        os.link(tmp, dest)  # refuse overwrites even on races
    finally:
        tmp.unlink(missing_ok=True)
    return {"path": str(dest), "sha256": digest(dest),
            "seconds": round(cursor, 3), "speaker": speaker, "style": style,
            "language": "ru", "model": MODEL_NAMES["voice"],
            "alignment": "measured_synthesis_chunk_boundaries_not_word_alignment",
            "subtitles": timeline, "human_voice_review": "pending"}


def _load_flux(model_dir: Path):
    try:
        import torch
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise OfflineBlocked("Install compatible torch/diffusers/transformers for local FLUX.2 Klein") from exc
    if not torch.cuda.is_available():
        raise OfflineBlocked("GPU_REQUIRED: FLUX.2 Klein has no tested CPU generation path here")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = Flux2KleinPipeline.from_pretrained(str(model_dir), torch_dtype=dtype,
                                             local_files_only=True)
    pipe.enable_model_cpu_offload()
    return pipe


@lru_cache(maxsize=1)
def flux_model(path: str):
    return _load_flux(Path(path))


def real_image(prompt: str, dest: Path, model_dir: Path, *, seed: int, width: int = 768,
               height: int = 1344, pipe=None, references: list[Path] | None = None) -> dict:
    if not isinstance(prompt, str) or len(prompt.strip()) < 30:
        raise OfflineBlocked("Missing detailed, approved scene prompt")
    if width % 16 or height % 16 or not 512 <= width <= 1536 or not 768 <= height <= 2048:
        raise ValueError("Unreasonable portrait image dimensions")
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    if pipe is None:
        pipe = flux_model(str(local_model(model_dir, "image")))
    from PIL import Image, ImageStat
    try:
        import torch
    except ImportError as exc:
        if pipe is None:
            raise OfflineBlocked("Torch is required for real GPU image inference") from exc
        # A pre-supplied pipeline may be a deterministic CPU-only test double;
        # real Flux GPU loading always imports torch at pipeline initialization.
        torch = None
    g = torch.Generator(device="cpu").manual_seed(seed) if torch else None
    kw = {"prompt": prompt, "width": width, "height": height,
          "num_inference_steps": 4, "guidance_scale": 1.0, "generator": g}
    provenance = []
    if references:
        if len(references) > 4:
            raise OfflineBlocked("FLUX.2 Klein reference count must be <=4")
        ref_images = []
        for ref in references:
            ref = Path(ref).resolve()
            if not ref.is_file():
                raise OfflineBlocked(f"Reference image missing: {ref.name}")
            with Image.open(ref) as opened:
                opened.verify()
            with Image.open(ref) as opened:
                ref_images.append(opened.convert("RGB").copy())
            provenance.append({"file": ref.name, "sha256": digest(ref)})
        kw["image"] = ref_images if len(ref_images) > 1 else ref_images[0]
    result = pipe(**kw)
    if not result.images:
        raise OfflineBlocked("FLUX produced no image")
    im = result.images[0].convert("RGB")
    if im.size != (width, height) or min(ImageStat.Stat(im).stddev) < 2:
        raise OfflineBlocked("FLUX image missing native resolution or nearly blank")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".image-", suffix=".png", dir=dest.parent,
                                     delete=False) as temp:
        tmp = Path(temp.name)
    try:
        im.save(tmp, format="PNG")
        with Image.open(tmp) as verify:
            verify.verify()
        os.link(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return {"path": str(dest), "sha256": digest(dest), "width": width, "height": height,
            "seed": seed, "model": MODEL_NAMES["image"],
            "human_visual_review": "pending", "character_consistency": "unverified",
            "reference_images": provenance}


def tone_score(dest: Path, seconds: float, mood: str = "sad", seed: int = 0,
               sample_rate: int = 24000) -> dict:
    """Original royalty-free-from-this-project ambient score; no samples or API.

    Deliberately restrained underscore; not a substitute for an expressive
    composed soundtrack. It is marked unreviewed for artistic quality.
    """
    import numpy as np
    profiles = {
        "sad": ((110.0, 164.81, 220.0), 0.080),
        "tender": ((174.61, 261.63, 349.23), 0.085),
        "mysterious": ((92.50, 138.59, 185.00), 0.065),
        "tense": ((98.0, 146.83, 155.56), 0.070),
        "hopeful": ((130.81, 196.0, 261.63), 0.085),
        "neutral": ((110.0, 164.81, 220.0), 0.065),
    }
    if mood not in profiles or not .1 <= seconds <= 600:
        raise ValueError("Invalid ambient mood or duration")
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    freqs, amplitude = profiles[mood]
    n = round(seconds * sample_rate)
    rng = np.random.default_rng(seed)
    # Constant-memory generation: synthesize in 1-second chunks to bound RAM.
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".score-", suffix=".wav", dir=dest.parent,
                                     delete=False) as f:
        tmp = Path(f.name)
    try:
        with wave.open(str(tmp), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            phase_shift = np.asarray(rng.uniform(-math.pi, math.pi, len(freqs)))
            written = 0
            while written < n:
                count = min(sample_rate, n - written)
                t = (np.arange(written, written+count, dtype=np.float64)/sample_rate)
                air = sum(np.sin(2 * math.pi * (hz * t + .02*np.sin(t*.25)) + phase_shift[i])
                          for i, hz in enumerate(freqs)) / len(freqs)
                swell = (.55 + .45 * np.sin(2 * math.pi * t / 9) ** 2)
                fade = np.minimum(1, np.minimum(t / .4, (seconds - t)/.5)).clip(0, 1)
                sample = np.clip(air*swell*fade*amplitude, -.98, .98)
                wav.writeframes((sample * 32767).astype("<i2").tobytes())
                written += count
        os.link(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return {"path": str(dest), "sha256": digest(dest), "seconds": n/sample_rate,
            "source": "original_procedural_score", "mood": mood,
            "artistic_review": "pending"}


def release_gpu_models() -> None:
    """Release model caches between visual and voice phases to avoid simultaneous OOM."""
    import gc
    flux_model.cache_clear()
    qwen_model.cache_clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass