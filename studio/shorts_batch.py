"""Prepare private existing Work stories as an honest production queue.

No generated visuals, invented subtitles, rewritten story or public export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def create(kit: Path, out: Path) -> dict:
    kit = kit.resolve()
    out = out.resolve()
    if not (kit / "stories_ru").is_dir() or not (kit / "narration_ru").is_dir():
        raise ValueError("Existing private Work kit must contain stories_ru and narration_ru")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Destination is non-empty; refuse overwriting private work")
    stories = sorted((kit / "stories_ru").glob("*.json"))
    if not stories:
        raise ValueError("No real story JSON files in private Work kit")
    jobs = []
    for source in stories:
        story = json.loads(source.read_text(encoding="utf-8"))
        key = str(story.get("key", ""))
        number = source.stem
        if not re.fullmatch(r"[0-9]{2}", number) or not key.startswith(number + "_"):
            raise ValueError("Story id mismatch")
        if story.get("lang") != "ru" or len(story.get("scenes", [])) < 8:
            raise ValueError(f"{number}: missing Russian scenes")
        matching = list((kit / "narration_ru").glob(f"{number}_*.txt"))
        if len(matching) != 1:
            raise ValueError(f"{number}: missing or ambiguous original Russian narration")
        src_text = matching[0]
        folder = out / number
        folder.mkdir(parents=True, exist_ok=False)
        target_text = folder / "story.txt"
        shutil.copyfile(src_text, target_text)
        scenes, shots = [], []
        for scene in story["scenes"]:
            idx = scene["id"]
            if not isinstance(idx, int) or idx < 1:
                raise ValueError("Invalid original scene number")
            scenes.append({
                "id": idx,
                "image": f"images/scene_{idx:02d}.png",
                "voice": f"voices/scene_{idx:02d}.wav",
                "subtitles": [],
            })
            shots.append({
                "scene": idx, "exact_source_text": scene["text"],
                "visual_prompt_ru": scene.get("visual_prompt_ru", ""),
                "music_direction_ru": scene.get("music_direction_ru", ""),
                "music_cue": scene.get("music_cue", ""),
                "status": "NOT_GENERATED",
            })
        project = {
            "schema": 1, "id": f"work_{number}", "title": story["title"],
            "language": "ru", "script_file": "story.txt",
            "script_sha256": hashlib.sha256(target_text.read_bytes()).hexdigest(),
            "width": 1080, "height": 1920, "fps": 24, "scenes": scenes,
        }
        (folder / "project.json").write_text(
            json.dumps(project, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        (folder / "shot_plan.json").write_text(
            json.dumps(shots, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        missing = {"images": len(scenes), "voices": len(scenes),
                   "aligned_subtitle_scenes": len(scenes),
                   "music_scene_beds": len(scenes)}
        jobs.append({"story_id": project["id"], "title": story["title"],
                     "scenes": len(scenes), "status": "ASSETS_PENDING",
                     "missing": missing})
    queue = {"schema": 1, "source": "private Work-authored Russian story kit",
             "status": "ASSETS_PENDING", "jobs": jobs,
             "note": "No real images or voices are present; no finished MP4 exists."}
    (out / "queue.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return queue


def main(argv=None):
    p = argparse.ArgumentParser(description="Prepare private Work story kit for real media")
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    print(json.dumps(create(args.kit, args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
