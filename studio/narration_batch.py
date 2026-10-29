"""Batch English narration with studio Kokoro-82M from private external scripts.

No scripts or media are embedded in the public repository.
No fallback robotic TTS, no publishing, no automatic release approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from studio.kokoro_tts import synthesize

_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,64}$")
_VOICE = re.compile(r"^[ab][fm]_[a-z0-9_]{2,40}$")


def prepare(manifest: Path, outdir: Path) -> list[dict]:
    base = manifest.resolve().parent
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not 1 <= len(data) <= 30:
        raise ValueError("Expected 1..30 batch entries")
    jobs = []
    seen = set()
    for row in data:
        key = row.get("id")
        voice = row.get("voice")
        language = row.get("language")
        source = row.get("text_file")
        if not isinstance(key, str) or not _ID.fullmatch(key) or key in seen:
            raise ValueError("Invalid or duplicate story id")
        if not isinstance(voice, str) or not _VOICE.fullmatch(voice) or language != voice[0]:
            raise ValueError(f"{key}: unsupported English voice preset")
        if not isinstance(source, str) or not re.fullmatch(r"[\w.-]{1,100}\.txt", source):
            raise ValueError(f"{key}: text_file must be a .txt filename")
        path = base / source
        if not path.is_file():
            raise FileNotFoundError(path)
        story = path.read_text(encoding="utf-8").strip()
        if not 50 <= len(story) <= 25000:
            raise ValueError(f"{key}: unexpected text length")
        jobs.append({
            "id": key, "voice": voice, "language": language, "text": story,
            "text_sha256": hashlib.sha256(story.encode()).hexdigest(),
            "output": str(outdir / f"{key}_{voice}.wav"),
        })
        seen.add(key)
    return jobs


def execute(jobs: list[dict], outdir: Path, dry_run: bool = False) -> dict:
    if dry_run:
        return {
            "dry_run": True,
            "jobs": [{k: v for k, v in job.items() if k != "text"} for job in jobs],
            "actual_kokoro_wavs": 0,
        }
    outdir.mkdir(parents=True, exist_ok=True)
    tracks = []
    for job in jobs:
        result = synthesize(
            job["text"], job["voice"], job["language"], Path(job["output"])
        )
        result.update({
            "id": job["id"], "voice": job["voice"],
            "language": job["language"], "text_sha256": job["text_sha256"],
        })
        tracks.append(result)
    receipt = outdir / "kokoro_batch_manifest.json"
    receipt.write_text(json.dumps({
        "schema": 1, "wav_tracks": tracks, "human_review": False
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "dry_run": False, "actual_kokoro_wavs": len(tracks),
        "manifest": str(receipt),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Narrate external English scripts with Kokoro")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    jobs = prepare(args.manifest, args.output_dir)
    print(json.dumps(execute(jobs, args.output_dir, args.dry_run), indent=2))


if __name__ == "__main__":
    main()
