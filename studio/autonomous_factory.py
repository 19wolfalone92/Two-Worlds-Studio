"""Resumable autonomous local-only scene pipeline for Work-approved Russian stories.

This is not a magical all-in-one model. It coordinates already-present weights
and existing offline renderer without third-party cloud APIs or subscriptions.
It never replaces unavailable real media with fake placeholders.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from .offline_media import (OfflineBlocked, accelerator_status, digest, local_model,
                            real_image, real_voice, tone_score)
from .shorts_factory import run as render_video, validate_project

MOOD_FROM_CUE = {
    "mystery": "mysterious", "suspense": "tense", "fear": "tense",
    "tense": "tense", "dark": "mysterious", "sad": "sad",
    "grief": "sad", "emotional": "sad", "love": "tender",
    "warm": "tender", "hope": "hopeful", "relief": "hopeful",
    "nostalgia": "tender", "detective": "mysterious",
}


def safe_child(root: Path, rel: str, ext: str) -> Path:
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise OfflineBlocked("Unsafe relative asset path")
    path = (root / rel).resolve()
    if root.resolve() not in path.parents or path.suffix.lower() != ext:
        raise OfflineBlocked("Asset escaped private project or has wrong extension")
    return path


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json",
                                     prefix=".writing-", dir=path.parent,
                                     delete=False) as f:
        tmp = Path(f.name)
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def inventory(project_dir: Path) -> dict:
    root = project_dir.resolve()
    p = json.loads((root / "project.json").read_text(encoding="utf-8"))
    shots = json.loads((root / "shot_plan.json").read_text(encoding="utf-8"))
    if len(shots) != len(p.get("scenes", [])) or len(shots) < 8:
        raise OfflineBlocked("Requires eight or more matching Work-approved scenes")
    script = root / p.get("script_file", "")
    if not script.is_file() or digest(script) != p.get("script_sha256"):
        raise OfflineBlocked("Original Work-approved script checksum changed")
    if [s["id"] for s in p["scenes"]] != [s["scene"] for s in shots]:
        raise OfflineBlocked("Original shot order changed")
    assembled = "\n\n".join([p["title"]] + [shot["exact_source_text"] for shot in shots])
    if assembled.strip() != script.read_text(encoding="utf-8").strip():
        raise OfflineBlocked("Scene narration differs from locked original story")
    scenes = []
    for s, shot in zip(p["scenes"], shots):
        img = safe_child(root, s["image"], ".png")
        voice = safe_child(root, s["voice"], ".wav")
        music_path = safe_child(root, s.get("music", f"music/scene_{s['id']:02d}.wav"), ".wav")
        scenes.append({"id": s["id"], "image": img.exists(), "voice": voice.exists(),
                       "music": music_path.exists(),
                       "subtitles": bool(s.get("subtitles")),
                       "shot_text_sha256": hashlib.sha256(
                           shot["exact_source_text"].encode("utf-8")).hexdigest()})
    missing = {key: sum(not s[key] for s in scenes)
               for key in ("image", "voice", "music", "subtitles")}
    return {"id": p["id"], "title": p["title"], "scenes": scenes,
            "missing": missing, "ready_for_video": not any(missing[x] for x in
            ("image", "voice", "subtitles")), "project_dir": str(root)}


def mood_of(shot: dict) -> str:
    cue = str(shot.get("music_cue", "") + " " + shot.get("music_direction_ru", "")).lower()
    for word, mood in MOOD_FROM_CUE.items():
        if word in cue:
            return mood
    return "neutral"


def run_story(project_dir: Path, out: Path, model_root: Path, *, images: bool,
              voices: bool, music: bool, render: bool, device: str = "cuda",
              speaker: str = "Serena", force_cpu_voice: bool = False,
              image_engine=real_image, voice_engine=real_voice,
              music_engine=tone_score, render_engine=render_video) -> dict:
    root = Path(project_dir).resolve()
    info = inventory(root)
    spec_path = root / "project.json"
    project = json.loads(spec_path.read_text(encoding="utf-8"))
    shots = json.loads((root / "shot_plan.json").read_text(encoding="utf-8"))
    failures = []
    made = {"image": 0, "voice": 0, "music": 0}
    gpu = accelerator_status()
    valid_voice = (device != "cuda" or gpu["cuda"] or force_cpu_voice)
    # Detect model readiness once. Never download weights or trigger paid APIs.
    image_ready = not images
    voice_ready = not voices
    if images:
        try:
            local_model(model_root, "image")
            if not gpu["cuda"]:
                raise OfflineBlocked("GPU_REQUIRED for offline FLUX image generation")
            image_ready = True
        except OfflineBlocked as exc:
            failures.append({"component": "images", "reason": str(exc)})
    if voices:
        try:
            local_model(model_root, "voice")
            if not valid_voice:
                raise OfflineBlocked("GPU_REQUIRED: set --device cpu explicitly to accept slower CPU voice")
            voice_ready = True
        except OfflineBlocked as exc:
            failures.append({"component": "voices", "reason": str(exc)})
    for scene, shot in zip(project["scenes"], shots):
        sid = scene["id"]
        image_path = safe_child(root, scene["image"], ".png")
        voice_path = safe_child(root, scene["voice"], ".wav")
        source_text = shot["exact_source_text"]
        if images and image_ready and not image_path.exists():
            try:
                seed = int(hashlib.sha256((project["id"] + str(sid)).encode()).hexdigest()[:8], 16)
                prompt = shot.get("visual_prompt_ru") or ""
                references = [safe_child(root, ref, ".png")
                              for ref in shot.get("reference_images", [])]
                image_engine(prompt, image_path, model_root, seed=seed,
                             references=references)
                made["image"] += 1
            except Exception as exc:
                failures.append({"scene": sid, "component": "image", "reason": str(exc)[:240]})
        if voices and voice_ready and not voice_path.exists():
            try:
                result = voice_engine(source_text, voice_path, model_root,
                                      speaker=speaker, style=mood_of(shot), device=device)
                scene["subtitles"] = result["subtitles"]
                atomic_json(spec_path, project)
                made["voice"] += 1
            except Exception as exc:
                failures.append({"scene": sid, "component": "voice", "reason": str(exc)[:240]})
        if music and voice_path.is_file():
            target = safe_child(root, scene.get("music", f"music/scene_{sid:02d}.wav"), ".wav")
            if not target.exists():
                try:
                    from .shorts_factory import duration as audio_duration
                    seconds = audio_duration(voice_path)
                    music_engine(target, seconds, mood=mood_of(shot), seed=sid)
                    made["music"] += 1
                except Exception as exc:
                    failures.append({"scene": sid, "component": "music", "reason": str(exc)[:240]})
            if target.exists() and scene.get("music") != str(target.relative_to(root)):
                scene["music"] = str(target.relative_to(root))
                atomic_json(spec_path, project)
    state = inventory(root)
    outcome = {"project": project["id"], "status": "ASSETS_READY" if state["ready_for_video"] else "ASSETS_PENDING",
               "created": made, "missing": state["missing"], "errors": failures,
               "gpu": gpu, "requires_paid_api": False, "offline": True}
    if render and state["ready_for_video"]:
        try:
            validate_project(spec_path)
            finished = render_engine(spec_path, out)
            outcome["status"] = "REVIEW_REQUIRED"
            outcome["render"] = finished
        except Exception as exc:
            outcome["status"] = "BLOCKED"
            outcome["errors"].append({"component": "render", "reason": str(exc)[:400]})
    elif render:
        outcome["errors"].append({"component": "render", "reason": "Real images, voices or synthesis-timed subtitles missing"})
    atomic_json(root / "local_factory_receipt.json", outcome)
    return outcome


def main(argv=None):
    p = argparse.ArgumentParser(description="Offline Two Worlds: real owned-model media + resumable MP4")
    p.add_argument("--project-dir", type=Path, required=True)
    p.add_argument("--models-root", type=Path, default=Path("private_models"))
    p.add_argument("--out", type=Path, default=Path("local_exports"))
    p.add_argument("--status", action="store_true")
    p.add_argument("--images", action="store_true")
    p.add_argument("--voices", action="store_true")
    p.add_argument("--music", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--speaker", default="Serena")
    args = p.parse_args(argv)
    try:
        if args.status:
            result = inventory(args.project_dir)
        else:
            result = run_story(args.project_dir, args.out, args.models_root,
                               images=args.images or args.all,
                               voices=args.voices or args.all,
                               music=args.music or args.all,
                               render=args.render or args.all,
                               device=args.device, speaker=args.speaker,
                               force_cpu_voice=(args.device == "cpu"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") != "BLOCKED" else 2
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())