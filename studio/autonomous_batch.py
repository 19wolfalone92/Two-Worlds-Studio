"""Private restartable Batch03 coordinator; never publishes or invents assets.

Uses existing locked Work stories 13..18 and studio.shorts_factory.
Image/TTS generation is delegated to a PRIVATE GPU runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

STORY_IDS = tuple(f"{n:02d}" for n in range(13, 19))
OUTPUT_FILES = ("final.mp4", "cover.jpg", "subtitles.srt", "qa_report.json", "receipt.json")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(base: Path, name: object, *, suffixes: tuple[str, ...]) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Missing or absolute private asset path")
    target = (base / name).resolve()
    if base.resolve() not in target.parents or target.suffix.lower() not in suffixes:
        raise ValueError("Unsafe private asset reference")
    return target


def _verified_cover(path: Path) -> bool:
    from PIL import Image
    try:
        with Image.open(path) as image:
            if image.width < 100 or image.height < 100:
                return False
            im = image.convert("RGB")
            im.thumbnail((90, 160))
            pixels = list(im.get_flattened_data() if hasattr(im, "get_flattened_data") else im.getdata())
            nearly_black = sum(max(rgb) < 18 for rgb in pixels)
            return bool(pixels) and nearly_black / len(pixels) < 0.55
    except (OSError, ValueError):
        return False


def _review_package(folder: Path, project: dict, source_hash: str) -> tuple[bool, str]:
    if folder.is_symlink() or not folder.is_dir():
        return False, "OUTPUT_MISSING"
    if any(not (folder / name).is_file() or (folder / name).is_symlink()
           for name in OUTPUT_FILES):
        return False, "OUTPUT_INCOMPLETE"
    try:
        receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
        qa = json.loads((folder / "qa_report.json").read_text(encoding="utf-8"))
        valid = (
            receipt.get("story_id") == project["id"]
            and receipt.get("script_sha256") == source_hash
            and receipt.get("final_sha256") == _hash(folder / "final.mp4")
            and receipt.get("review_status") == "REVIEW_REQUIRED"
            and receipt.get("approved_for_publication") is False
            and qa.get("technical_pass") is True
            and qa.get("full_decode_pass") is True
            and qa.get("approved_for_publication") is False
            and qa.get("black_intervals") == []
            and isinstance(qa.get("first_frame_mean"), (int, float))
            and qa["first_frame_mean"] >= 9
            and _verified_cover(folder / "cover.jpg")
        )
    except (OSError, ValueError, TypeError, KeyError):
        valid = False
    return (True, "REVIEW_REQUIRED") if valid else (False, "INVALID_OR_BLACK_OUTPUT")


def inspect_story(queue_root: Path, export_root: Path, number: str) -> dict:
    if number not in STORY_IDS:
        raise ValueError("Unknown Batch03 story ID")
    answer = {"story": number, "status": "WAITING_FOR_PRIVATE_SOURCE",
              "next_action": "ATTACH_PRIVATE_BATCH03_KIT", "missing": {}}
    folder = queue_root / number
    if folder.is_symlink() or not folder.is_dir() or queue_root.resolve() not in folder.resolve().parents:
        return answer
    manifest = folder / "project.json"
    if manifest.is_symlink() or not manifest.is_file():
        return answer
    try:
        project = json.loads(manifest.read_text(encoding="utf-8"))
        if (project.get("id") != f"work_{number}" or project.get("schema") != 1
                or project.get("language") != "ru" or not isinstance(project.get("scenes"), list)
                or len(project["scenes"]) < 8):
            raise ValueError("Unexpected Russian story manifest")
        script = _contained(folder, project.get("script_file"), suffixes=(".txt",))
        if not script.is_file():
            raise ValueError("Locked original narration missing")
        original_sha = _hash(script)
        if original_sha != project.get("script_sha256"):
            raise ValueError("Original narration SHA256 mismatch")
        missing = {"images": 0, "voices": 0, "subtitles": 0}
        seen = set()
        for scene in project["scenes"]:
            if not isinstance(scene, dict) or type(scene.get("id")) is not int or scene["id"] in seen:
                raise ValueError("Invalid or repeated scene")
            seen.add(scene["id"])
            image = _contained(folder, scene.get("image"), suffixes=(".jpg", ".jpeg", ".png", ".webp"))
            voice = _contained(folder, scene.get("voice"), suffixes=(".wav", ".flac", ".mp3", ".m4a", ".ogg"))
            missing["images"] += not image.is_file()
            missing["voices"] += not voice.is_file()
            missing["subtitles"] += not isinstance(scene.get("subtitles"), list) or not scene["subtitles"]
        answer["missing"] = missing
        if missing["images"]:
            answer.update(status="ASSETS_PENDING", next_action="GENERATE_IMAGES_PRIVATE_GPU")
        elif missing["voices"]:
            answer.update(status="ASSETS_PENDING", next_action="GENERATE_RUSSIAN_VOICE_PRIVATE_GPU")
        elif missing["subtitles"]:
            answer.update(status="ASSETS_PENDING", next_action="ALIGN_VOICE_TO_SUBTITLES")
        else:
            from .shorts_factory import validate_project
            validate_project(manifest)
            answer.update(status="READY_TO_RENDER", next_action="RENDER_OFFLINE_CPU")
        output = export_root / f"work_{number}"
        if output.exists() or output.is_symlink():
            verified, why = _review_package(output, project, original_sha)
            if verified and not any(missing.values()):
                answer.update(status="REVIEW_REQUIRED", next_action="HUMAN_REVIEW_BEFORE_UPLOAD")
            else:
                answer.update(status="OUTPUT_BLOCKED", next_action=why)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        answer.update(status="INPUT_BLOCKED", next_action="REPAIR_PRIVATE_PROJECT",
                      error=type(exc).__name__)
    return answer


def plan(queue_root: Path, export_root: Path) -> dict:
    stories = [inspect_story(queue_root, export_root, number) for number in STORY_IDS]
    counts = {key: sum(s["status"] == key for s in stories) for key in
              ("WAITING_FOR_PRIVATE_SOURCE", "ASSETS_PENDING", "READY_TO_RENDER",
               "REVIEW_REQUIRED", "INPUT_BLOCKED", "OUTPUT_BLOCKED")}
    return {"schema": 1, "batch": "RU_BATCH03_13_18", "expected_stories": 6,
            "technical_exports_ready_for_review": counts["REVIEW_REQUIRED"],
            "approved_for_publication": False, "published": 0,
            "counts": counts, "stories": stories}


def execute(queue_root: Path, export_root: Path, *, max_jobs: int = 1) -> dict:
    if not 1 <= max_jobs <= 6:
        raise ValueError("max_jobs must be between 1 and 6")
    from .shorts_factory import run
    export_root.mkdir(parents=True, exist_ok=True)
    executed = []
    for item in plan(queue_root, export_root)["stories"]:
        if len(executed) >= max_jobs:
            break
        if item["status"] != "READY_TO_RENDER":
            continue
        try:
            result = run(queue_root / item["story"] / "project.json", export_root)
            executed.append({"story": item["story"], "status": result["status"]})
        except (OSError, ValueError, RuntimeError) as exc:
            executed.append({"story": item["story"], "status": "RENDER_FAILED",
                             "error": type(exc).__name__})
    report = plan(queue_root, export_root)
    report["render_attempts"] = executed
    return report


def atomic_report(path: Path, report: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".batch-status-",
                                     suffix=".json", encoding="utf-8", delete=False) as file:
        tmp = Path(file.name)
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(tmp, path)


def watch(queue_root: Path, export_root: Path, *, max_jobs: int = 1,
          interval_seconds: int = 600, max_cycles: int = 0,
          report_path: Path | None = None) -> dict:
    """Optional persistent private CPU coordinator; 0 cycles = until stopped."""
    if not 30 <= interval_seconds <= 86400 or max_cycles < 0:
        raise ValueError("Invalid polling interval/cycle count")
    cycles = 0
    last = {}
    while max_cycles == 0 or cycles < max_cycles:
        last = execute(queue_root, export_root, max_jobs=max_jobs)
        if report_path is not None:
            atomic_report(report_path, last)
        cycles += 1
        if last["technical_exports_ready_for_review"] == len(STORY_IDS):
            break
        if max_cycles and cycles >= max_cycles:
            break
        time.sleep(interval_seconds)
    return last


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Private restartable TWO WORLDS Batch03 (no publication)")
    parser.add_argument("--queue-root", type=Path, required=True, help="PRIVATE extracted Work queue")
    parser.add_argument("--out", type=Path, required=True, help="PRIVATE media output")
    parser.add_argument("--mode", choices=("status", "render", "watch"), default="status")
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--max-cycles", type=int, default=0, help="watch: 0 means until stopped")
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--report", type=Path, help="optional PRIVATE JSON status file")
    args = parser.parse_args(argv)
    if args.mode == "watch":
        output = watch(args.queue_root, args.out, max_jobs=args.max_jobs,
                       interval_seconds=args.interval_seconds, max_cycles=args.max_cycles,
                       report_path=args.report)
    else:
        output = (execute(args.queue_root, args.out, max_jobs=args.max_jobs)
                  if args.mode == "render" else plan(args.queue_root, args.out))
        if args.report:
            atomic_report(args.report, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if not any(row["status"] == "RENDER_FAILED" for row in output.get("render_attempts", [])) else 2


if __name__ == "__main__":
    raise SystemExit(main())
