"""TWO WORLDS V6: identity-first, genuine I2V-only production (private media).

The old 7-independent-SDXL-shot EB01 preview is NOT accepted production art.
This orchestrator combines existing official-model ComfyUI integration, locked
character refs, visual/motion approval, and native Studio mastering. No heavy
model weights are vendored; no GPU is run without an explicit --execute-local-gpu
flag. No automatic publishing or fabricated approval occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

from PIL import Image, ImageStat

MODEL = "wan22-5b"
CAST = ("elizabeth", "kostya")
SCENES = ("S00", "S01", "S02", "S03", "S04", "S05", "S06")
LINE_LOCK = {
    "S01": "Следующий. Имя?",
    "S02": "Константин Мортисович Костин. Для друзей — Костя.",
    "S03": "Возраст?",
    "S04": "Двести три. Но душой я мальчишка!",
    "S05": "Вижу. А где душа?",
    "S06": "На испытательном. Уже три века.",
}
SCENE_CAST = {
    "S00": {"elizabeth", "kostya"}, "S01": {"elizabeth"},
    "S02": {"kostya"}, "S03": {"elizabeth"}, "S04": {"kostya"},
    "S05": {"elizabeth"}, "S06": {"kostya"},
}
VISUAL = ("identity", "wardrobe", "anatomy", "lighting", "composition",
          "kostya_is_skeleton")
MOTION = ("identity_stable", "body_acting", "jaw_and_cheeks_coherent",
          "wardrobe_stable", "natural_temporal_motion")

class ProductionBlocked(ValueError):
    """Hard stop before a new expensive generation or bad export."""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict:
    if path.stat().st_size > 256_000:
        raise ProductionBlocked("Oversized private production manifest")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ProductionBlocked("Manifest root must be an object")
    return obj


def private_root(path: Path) -> Path:
    root = path.resolve(strict=True)
    source_tree = Path(__file__).resolve().parent.parent
    if not root.is_dir() or root == source_tree or source_tree in root.parents:
        raise ProductionBlocked("Private source art cannot be written into public Git tree")
    return root


def asset(root: Path, relative: str, extensions: tuple[str, ...]) -> Path:
    if not isinstance(relative, str) or len(relative) > 240 or not relative or "\\" in relative:
        raise ProductionBlocked("Bad relative private asset name")
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or p.suffix.lower() not in extensions:
        raise ProductionBlocked("Invalid private media path/type")
    target = root.joinpath(p).resolve()
    if root not in target.parents or target.is_symlink() or not target.is_file():
        raise ProductionBlocked("Expected private asset missing or symlinked: " + relative)
    return target


def png(root: Path, relative: str, expected_sha: str) -> Path:
    p = asset(root, relative, (".png",))
    if not re.fullmatch("[a-f0-9]{64}", str(expected_sha)) or digest(p) != expected_sha:
        raise ProductionBlocked("Image changed from locked canonical identity")
    with Image.open(p) as img:
        if img.format != "PNG" or img.width < 576 or img.height < 1024:
            raise ProductionBlocked("Approved reference/keyframe has insufficient resolution")
        stats = ImageStat.Stat(img.convert("RGB"))
        if max(stats.stddev) < 8.0:
            raise ProductionBlocked("Blank or near-blank reference image")
    return p


def read_storyboard(path: Path) -> dict:
    story = _json(path)
    if story.get("schema") != 6 or story.get("episode") != "ELIZABETH_BONES_E01":
        raise ProductionBlocked("V6 expects the locked Elizabeth E01 storyboard")
    if story.get("model") != MODEL or story.get("fps") != 24:
        raise ProductionBlocked("V6 uses original official Wan2.2 5B I2V at 24 FPS")
    refs = story.get("references")
    if not isinstance(refs, dict) or set(refs) != set(CAST):
        raise ProductionBlocked("Both canonical subject references are required")
    shots = story.get("shots")
    if not isinstance(shots, list) or [x.get("id") for x in shots if isinstance(x, dict)] != list(SCENES):
        raise ProductionBlocked("Exactly seven locked scenes in production order are required")
    for shot in shots:
        sid = shot["id"]
        if set(shot.get("characters", [])) != SCENE_CAST[sid]:
            raise ProductionBlocked("Wrong required cast in " + sid)
        if shot.get("caption") != LINE_LOCK.get(sid, ""):
            raise ProductionBlocked("Incorrect six-line locked Russian script: " + sid)
        secs = shot.get("duration")
        if type(secs) not in (int, float) or not math.isfinite(secs) or not 1 <= secs <= 15:
            raise ProductionBlocked("Every shot needs a bounded duration")
        if not isinstance(shot.get("motion_prompt"), str) or not 18 <= len(shot["motion_prompt"]) <= 1600:
            raise ProductionBlocked("Real actor motion instructions missing: " + sid)
        if sid != "S00" and (not shot.get("voice") or not shot.get("voice_sha256")):
            raise ProductionBlocked("Every scripted line needs original real audio " + sid)
    return story


def checked_visuals(root: Path, story_path: Path, review_path: Path) -> tuple[dict, dict, dict]:
    story = read_storyboard(story_path)
    review = _json(review_path)
    if (review.get("schema") != 6 or review.get("episode") != story["episode"]
            or review.get("storyboard_sha256") != digest(story_path)
            or review.get("human_visual_review") is not True
            or review.get("approved") is not True
            or review.get("reviewer") not in {"owner", "appointed_human"}):
        raise ProductionBlocked("Visual storyboard was not explicitly human-reviewed or changed after approval")
    canon = {}
    for who in CAST:
        row = story["references"][who]
        canon[who] = png(root, row["image"], row["sha256"])
        if review.get("reference_approved", {}).get(who) is not True:
            raise ProductionBlocked("Canonical subject rejected or missing human approval: " + who)
    frames = {}
    for shot in story["shots"]:
        sid = shot["id"]
        frames[sid] = png(root, shot["image"], shot["image_sha256"])
        checks = review.get("visual_checks", {}).get(sid, {})
        if not isinstance(checks, dict) or any(checks.get(key) is not True for key in VISUAL):
            raise ProductionBlocked("Image identity/anatomy/wardrobe review rejected: " + sid)
        if sid != "S00":
            sound = asset(root, shot["voice"], (".wav", ".mp3", ".flac", ".m4a"))
            if digest(sound) != shot["voice_sha256"] or sound.stat().st_size < 4096:
                raise ProductionBlocked("Dialogue source changed or missing in " + sid)
    return story, review, frames


def check_clip(root: Path, sid: str, row: dict, frame: Path, length: float) -> tuple[Path, dict]:
    clip = asset(root, row.get("clip"), (".mp4",))
    receipt = asset(root, row.get("receipt"), (".json",))
    if digest(clip) != row.get("sha256"):
        raise ProductionBlocked("Generated neural video bytes changed: " + sid)
    proof = _json(receipt)
    if (proof.get("engine") != "neural_comfy" or proof.get("model") != MODEL
            or proof.get("clip_sha256") != digest(clip)
            or proof.get("reference_sha256") != digest(frame)
            or proof.get("inference_performed") is not True
            or proof.get("published") is not False):
        raise ProductionBlocked("No genuine source-locked Comfy Wan I2V proof " + sid)
    # FFprobe/decoding alone once accepted a broken 49-frame clip whose last
    # three frames contained neon streaks. Reject temporal color corruption
    # BEFORE allowing the I2V take into any final native editor.
    from studio.eb01_motion_artifact_gate import inspect_clip
    artifact_audit = inspect_clip(clip)
    if artifact_audit["status"] != "NO_NEON_ARTIFACT_FOUND":
        raise ProductionBlocked(
            "Neural actor video rejected for neon/temporal artifacts in "
            + sid + ": " + artifact_audit["status"]
        )
    from studio.comfy_neural import import_take
    checked = import_take(clip, receipt, frame, duration=length)
    checked["temporal_neon_artifact_qc"] = artifact_audit["status"]
    return clip, checked


def plan(story_path: Path) -> dict:
    story = read_storyboard(story_path)
    return {"status": "PLAN_ONLY_NO_GPU", "engine": MODEL,
            "editor": "studio.hybrid_master.produce",
            "storyboard_sha256": digest(story_path),
            "scenes": len(story["shots"]), "spoken_lines": len(LINE_LOCK),
            "image_method": "identity_conditioned_from_approved_canon",
            "media_source": "private_only", "requires_human_visual_review": True,
            "requires_real_i2v_pilot_and_full_clip_review": True,
            "neural_inference_performed": False, "published": False}


def pilot(story_path: Path, review_path: Path, root: Path, shot_id: str,
          *, execute: bool, accept_license: bool, host: str) -> dict:
    root = private_root(root)
    story, _, frames = checked_visuals(root, story_path, review_path)
    shot = next((s for s in story["shots"] if s["id"] == shot_id), None)
    if shot is None or shot_id not in ("S01", "S02"):
        raise ProductionBlocked("Pilot must be one approved Elizabeth or Kostya speaking shot")
    output = root / "clips" / (shot_id + ".mp4")
    if execute and not accept_license:
        raise ProductionBlocked("A real video GPU inference requires reviewed model licence")
    if not execute:
        return {"status": "PILOT_READY_NO_GPU", "shot": shot_id, "image_sha256": digest(frames[shot_id]),
                "backend": MODEL, "output": output.name, "gpu_executed": False, "published": False}
    if output.exists() or output.with_suffix(".receipt.json").exists():
        raise ProductionBlocked("Never overwrite a reviewed or existing neural take")
    # One actual short video candidate only, 49 frames = 2.04 s at 24 FPS.
    # No hidden retry, multi-seed fanout, cloud GPU job or auto-publishing.
    output.parent.mkdir(parents=True, exist_ok=True)
    from studio.comfy_neural import generate
    result = generate(MODEL, reference=frames[shot_id], output=output,
                      prompt=shot["motion_prompt"], seed=42, host=host,
                      width=576, height=1024, frames=49, steps=25,
                      timeout=3600, execute=True, accepted_license=True)
    return {**result, "human_motion_review_required": True, "next_stage": "review_pilot_before_full_generation",
            "published": False}


def render(story_path: Path, review_path: Path, takes_path: Path,
           root: Path, output_name: str) -> dict:
    root = private_root(root)
    story, review, frames = checked_visuals(root, story_path, review_path)
    takes = _json(takes_path)
    if (takes.get("schema") != 6 or takes.get("storyboard_sha256") != digest(story_path)
            or set(takes.get("shots", {})) != set(SCENES)):
        raise ProductionBlocked("Seven full real neural video takes required; no static frame fallback")
    motion = review.get("motion_checks", {})
    if set(motion) != set(SCENES) or review.get("pilot_approved") is not True:
        raise ProductionBlocked("Human-approved motion pilot and every animated shot required")
    locked = []
    checked_sha = {}
    for shot in story["shots"]:
        sid = shot["id"]
        row = takes["shots"][sid]
        clip, _ = check_clip(root, sid, row, frames[sid], shot["duration"])
        checks = motion.get(sid, {})
        if (not isinstance(checks, dict) or checks.get("sha256") != digest(clip)
                or checks.get("reviewer") not in {"owner", "appointed_human"}
                or checks.get("approved") is not True
                or any(checks.get(key) is not True for key in MOTION)):
            raise ProductionBlocked("Missing real temporal performance approval: " + sid)
        # Lock source-pinned neural take in native hybrid editor (never Ken Burns).
        manifest_shot = {
            "clip": row["clip"], "duration": shot["duration"], "shot_type": "medium",
            "camera": "push_in", "camera_strength": 0.,
            "caption": shot["caption"], "caption_position": "bottom",
            "approved": True, "clip_take": {
                "engine": "neural_comfy", "visual_profile": "photoreal_cinematic",
                "sha256": row["sha256"], "receipt": row["receipt"], "reference": shot["image"]
            },
        }
        if sid != "S00":
            manifest_shot.update(voice=shot["voice"], dialogue_text=shot["caption"],
                                 voice_offset=0.08, voice_bus="dialogue", voice_approved=True)
        locked.append(manifest_shot)
        checked_sha[sid] = digest(clip)
    if (not re.fullmatch(r"[A-Za-z0-9_.-]{5,100}\.mp4", output_name)
            or output_name.startswith(".")):
        raise ProductionBlocked("Output filename must be a private regular MP4")
    movie = root / "renders" / output_name
    if movie.exists() or movie.is_symlink():
        raise ProductionBlocked("Do not overwrite old preview or finished work")
    movie.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1, "delivery_profile": "mobile",
        "language": "ru", "visual_profile": "photoreal_cinematic",
        "editorial_approved": False, "rights_approved": False,
        "burn_subtitles": True, "shots": locked
    }
    manifest_path = root / "v6_approved_clips_manifest.json"
    if manifest_path.exists():
        raise ProductionBlocked("Existing render manifest would be overwritten")
    with manifest_path.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    from studio.hybrid_master import produce
    report = produce(manifest_path, movie, master=False, quality_check=True)
    if report.get("shots") != len(SCENES) or report.get("static_camera_shots") or report.get("motion_clips") != len(SCENES):
        raise ProductionBlocked("Native Studio attempted to use static frames or missed clips")
    proof = {"schema": 6, "status": "RENDERED_REVIEW_ONLY",
             "editor": "studio.hybrid_master.produce", "invideo": False,
             "real_clip_sha256": checked_sha, "movie_sha256": digest(movie),
             "movie": str(movie), "visual_review_sha256": digest(review_path),
             "human_final_watch_passed": False, "rights_approved": False,
             "published": False}
    with (root / "renders" / (movie.stem + ".v6-provenance.json")).open("x", encoding="utf-8") as f:
        json.dump(proof, f, ensure_ascii=False, indent=2)
    return proof


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="TWO WORLDS V6 real I2V / never random SDXL montage")
    commands = p.add_subparsers(dest="action", required=True)
    a = commands.add_parser("plan")
    a.add_argument("--storyboard", type=Path, required=True)
    b = commands.add_parser("pilot")
    b.add_argument("--storyboard", type=Path, required=True)
    b.add_argument("--assets", type=Path, required=True)
    b.add_argument("--review", type=Path, required=True)
    b.add_argument("--shot", choices=("S01", "S02"), default="S02")
    b.add_argument("--host", default="http://127.0.0.1:8188")
    b.add_argument("--execute-local-gpu", action="store_true")
    b.add_argument("--accept-model-license", action="store_true")
    c = commands.add_parser("render")
    c.add_argument("--storyboard", type=Path, required=True)
    c.add_argument("--review", type=Path, required=True)
    c.add_argument("--takes", type=Path, required=True)
    c.add_argument("--assets", type=Path, required=True)
    c.add_argument("--name", default="ELIZABETH_BONES_E01_REVIEW_ONLY.mp4")
    x = p.parse_args(argv)
    try:
        if x.action == "plan":
            result = plan(x.storyboard)
        elif x.action == "pilot":
            result = pilot(x.storyboard, x.review, x.assets, x.shot,
                           execute=x.execute_local_gpu,
                           accept_license=x.accept_model_license, host=x.host)
        else:
            result = render(x.storyboard, x.review, x.takes, x.assets, x.name)
    except (ProductionBlocked, FileNotFoundError, json.JSONDecodeError) as exc:
        result = {"status": "BLOCKED", "reason": str(exc),
                  "gpu_executed": False, "published": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
