"""Russian neural narration for TWO WORLDS Studio (optional Qwen3-TTS backend).

Only renders explicit, already-written text. Does NOT translate or edit scripts,
clone voices, mix media, approve output or publish anything.
Model files and input scripts remain outside the public repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

MODEL_SMALL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODEL_LARGE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
ALLOWED_MODELS = (MODEL_SMALL, MODEL_LARGE)
VOICES = ("Serena", "Ryan", "Vivian", "Aiden", "Sohee", "Ono_Anna",
          "Dylan", "Eric", "Uncle_Fu")
STYLE_PROMPTS = {
    "neutral": "",
    "warm": "Speak warmly, sincerely and naturally, like a thoughtful storyteller.",
    "sad": "Speak softly with genuine sadness and restrained emotion; avoid melodrama.",
    "mysterious": "Narrate quietly and mysteriously, with suspenseful natural pauses.",
    "tense": "Speak with controlled tension and urgency; keep articulation clear.",
    "joyful": "Speak with genuine, light happiness and natural energy.",
}
MAX_TEXT = 25000
SAMPLE_LIMIT_SECONDS = 600


def validate(text: str, voice: str, style: str, model_id: str) -> None:
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= MAX_TEXT:
        raise ValueError("Russian narration text must be 1..25000 characters")
    if not re.search(r"[А-Яа-яЁё]", text):
        raise ValueError("Russian narration requires Cyrillic text; no automatic translation")
    if voice not in VOICES:
        raise ValueError(f"Unsupported Qwen Russian voice: {voice!r}")
    if style not in STYLE_PROMPTS:
        raise ValueError(f"Unknown emotion/style: {style!r}")
    if model_id not in ALLOWED_MODELS:
        raise ValueError("Only approved Qwen3-TTS CustomVoice model IDs are supported")


def split_text(text: str, limit: int = 350) -> list[str]:
    """Split prose at sentence boundaries, then words, without dropping text."""
    if not isinstance(text, str) or not text.strip() or limit < 30:
        raise ValueError("Invalid text or chunk limit")
    normalized = re.sub(r"\s+", " ", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?…])\s+", normalized):
        if not sentence:
            continue
        for word in sentence.split(" "):
            if len(word) > limit:
                raise ValueError("A word is longer than the synthesis chunk limit")
            attempt = f"{current} {word}".strip() if current else word
            if len(attempt) > limit:
                if current:
                    chunks.append(current)
                current = word
            else:
                current = attempt
    if current:
        chunks.append(current)
    if not chunks or len(chunks) > 120:
        raise ValueError("Empty or excessively fragmented narration")
    return chunks


@lru_cache(maxsize=2)
def _get_model(model_id: str):
    """Lazily load model on an authorized GPU/CPU runtime; no CI downloads."""
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError(
            "Russian TTS requires optional qwen-tts, torch, numpy and soundfile. "
            "Install them in a separate GPU environment; dry-run needs none."
        ) from exc
    cuda = torch.cuda.is_available()
    return Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0" if cuda else "cpu",
        dtype=torch.float16 if cuda else torch.float32,
    )


def synthesize(
    text: str,
    voice: str,
    output: Path,
    *,
    style: str = "neutral",
    model_id: str = MODEL_SMALL,
) -> dict:
    """Generate verified WAV with no fake voice, no silent fallback, no overwrites."""
    validate(text, voice, style, model_id)
    output = Path(output)
    if output.suffix.lower() != ".wav":
        raise ValueError("Russian voice output must be a WAV file")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite voice: {output}")
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Russian TTS requires numpy and soundfile") from exc
    model = _get_model(model_id)
    blocks = []
    sr = None
    for part in split_text(text):
        wavs, rate = model.generate_custom_voice(
            text=part, language="Russian", speaker=voice,
            instruct=STYLE_PROMPTS[style],
        )
        if not isinstance(rate, int) or not 8000 <= rate <= 96000 or not wavs:
            raise RuntimeError("Qwen3-TTS returned invalid audio sample rate/data")
        if sr is None:
            sr = rate
        elif sr != rate:
            raise RuntimeError("Inconsistent sample rate between chunks")
        samples = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if not len(samples) or not np.isfinite(samples).all():
            raise RuntimeError("Qwen3-TTS returned empty or non-finite audio")
        if blocks:
            blocks.append(np.zeros(int(rate * 0.2), dtype=np.float32))
        blocks.append(samples)
    pcm = np.concatenate(blocks)
    duration = len(pcm) / sr
    if not 0.05 <= duration <= SAMPLE_LIMIT_SECONDS:
        raise ValueError("Voice audio length is outside the permitted 0.05..600s")
    # Avoid any partial/truncated file replacing an existing studio asset.
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".qwen-ru-", suffix=".wav", dir=output.parent)
    os.close(fd)
    try:
        sf.write(tmp, np.clip(pcm, -1, 1), sr, format="WAV", subtype="PCM_16")
        with sf.SoundFile(tmp) as f:
            if f.frames != len(pcm) or f.samplerate != sr or f.channels != 1:
                raise RuntimeError("Rendered WAV failed structural validation")
        os.link(tmp, output)  # atomic creation; fails rather than overwriting
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return {
        "path": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "sample_rate": sr,
        "duration_seconds": round(duration, 3),
        "model": model_id,
        "engine": "qwen3-tts",
        "language": "ru",
        "voice": voice,
        "style": style,
        "review_status": "unreviewed",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Russian narration for Work-approved text; no script editing"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Explicit Russian text to narrate")
    group.add_argument("--text-file", type=Path, help="Private UTF-8 Russian script")
    parser.add_argument("--voice", choices=VOICES, default="Serena")
    parser.add_argument("--style", choices=tuple(STYLE_PROMPTS), default="warm")
    parser.add_argument("--model", choices=ALLOWED_MODELS, default=MODEL_SMALL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    validate(text, args.voice, args.style, args.model)
    if args.dry_run:
        print(json.dumps({
            "dry_run": True, "language": "ru", "voice": args.voice,
            "style": args.style, "model": args.model,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "planned_chunks": len(split_text(text)), "actual_audio_clips": 0,
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(
            synthesize(text, args.voice, args.output, style=args.style, model_id=args.model),
            ensure_ascii=False, indent=2,
        ))


if __name__ == "__main__":
    main()
