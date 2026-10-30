"""Private English/Russian narration batch. Never writes or translates stories."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

from studio.kokoro_tts import synthesize as kokoro_synthesize
from studio.russian_qwen_tts import (
    MODEL_SMALL, synthesize as russian_synthesize, validate as validate_russian,
)

_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,64}$")
_EN_VOICE = re.compile(r"^[ab][fm]_[a-z0-9_]{2,40}$")
_SAFE_TEXT = re.compile(r"[A-Za-z0-9_.-]{1,100}\.txt")


def prepare(manifest: Path, outdir: Path) -> list[dict]:
    base = manifest.resolve().parent
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not 1 <= len(data) <= 30:
        raise ValueError("Expected 1..30 batch entries")
    jobs = []
    seen = set()
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("Every entry must be an object")
        key, voice, language, source = (row.get(x) for x in ("id", "voice", "language", "text_file"))
        style = row.get("style", "neutral")
        if not isinstance(key, str) or not _ID.fullmatch(key) or key in seen:
            raise ValueError("Invalid or duplicate story id")
        if not isinstance(source, str) or not _SAFE_TEXT.fullmatch(source):
            raise ValueError(f"{key}: text_file must be a safe .txt filename")
        path = (base / source).resolve()
        if path.parent != base or not path.is_file():
            raise ValueError(f"{key}: missing or unsafe text file")
        story = path.read_text(encoding="utf-8").strip()
        if not 50 <= len(story) <= 25000:
            raise ValueError(f"{key}: unexpected text length")
        model = row.get("model", MODEL_SMALL)
        if language == "ru":
            if not isinstance(voice, str) or not voice.startswith("qwen:"):
                raise ValueError(f"{key}: Russian voice must use qwen:<speaker>")
            speaker = voice.split(":", 1)[1]
            validate_russian(story, speaker, style, model)
            engine, filevoice = "qwen3-tts", f"qwen_{speaker}"
        elif (isinstance(voice, str) and _EN_VOICE.fullmatch(voice)
              and language == voice[0] and style == "neutral" and "model" not in row):
            engine, filevoice, model = "kokoro", voice, None
        else:
            raise ValueError(f"{key}: unsupported language, voice or style")
        jobs.append({
            "id": key, "voice": voice, "language": language, "text": story,
            "style": style, "model": model, "engine": engine,
            "text_sha256": hashlib.sha256(story.encode()).hexdigest(),
            "output": str(outdir / f"{key}_{filevoice}.wav"),
        })
        seen.add(key)
    return jobs


def execute(jobs: list[dict], outdir: Path, dry_run: bool = False) -> dict:
    if dry_run:
        return {
            "dry_run": True, "jobs": [
                {k: v for k, v in job.items() if k != "text"} for job in jobs
            ], "actual_kokoro_wavs": 0, "actual_qwen_wavs": 0,
        }
    outdir.mkdir(parents=True, exist_ok=True)
    tracks, counts = [], {"kokoro": 0, "qwen3-tts": 0}
    # Refuse partial reruns before creating any WAV.
    if any(Path(job["output"]).exists() for job in jobs):
        raise FileExistsError("A target WAV already exists")
    for job in jobs:
        if job["engine"] == "qwen3-tts":
            result = russian_synthesize(
                job["text"], job["voice"].split(":", 1)[1], Path(job["output"]),
                style=job["style"], model_id=job["model"],
            )
        elif job["engine"] == "kokoro":
            result = kokoro_synthesize(
                job["text"], job["voice"], job["language"], Path(job["output"])
            )
        else:
            raise ValueError("Unknown narration engine")
        counts[job["engine"]] += 1
        result.update({
            "id": job["id"], "voice": job["voice"], "language": job["language"],
            "text_sha256": job["text_sha256"], "engine": job["engine"],
        })
        tracks.append(result)
    english_only = all(j["engine"] == "kokoro" for j in jobs)
    receipt = outdir / (
        "kokoro_batch_manifest.json" if english_only else "narration_batch_manifest.json"
    )
    if receipt.exists():
        raise FileExistsError("Refusing to overwrite narration manifest")
    receipt.write_text(json.dumps({
        "schema": 1, "wav_tracks": tracks, "human_review": False
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "dry_run": False, "actual_kokoro_wavs": counts["kokoro"],
        "actual_qwen_wavs": counts["qwen3-tts"], "manifest": str(receipt),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Narrate private English/Russian scripts")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(execute(
        prepare(args.manifest, args.output_dir), args.output_dir, args.dry_run
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
