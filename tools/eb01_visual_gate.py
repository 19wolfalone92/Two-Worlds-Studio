"""Fail-closed visual approval before EB01 GPU animation/render.

CPU-only: this does not generate images, call a model, edit a video or spend GPU.
The checks prove media structure / hashes, NOT character semantics. Human visual
inspection of the contact sheet is mandatory and explicitly recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageStat

SCENES = ("S00", "S01", "S02", "S03", "S04", "S05", "S06")
CAST = {
    "S00": {"elizabeth", "kostya"},
    "S01": {"elizabeth"},
    "S02": {"kostya"},
    "S03": {"elizabeth"},
    "S04": {"kostya"},
    "S05": {"elizabeth"},
    "S06": {"kostya"},
}
CHECKS = ("identity", "costume", "anatomy", "composition", "lighting", "no_human_instead_of_skeleton")

class ReviewBlocked(ValueError):
    pass

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def safely_resolve(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or name.startswith("/"):
        raise ReviewBlocked("Invalid asset path")
    dest = (root / name).resolve()
    if dest == root.resolve() or root.resolve() not in dest.parents or dest.is_symlink():
        raise ReviewBlocked("Media must be regular files under the review folder")
    if not dest.is_file():
        raise ReviewBlocked("Missing media " + name)
    return dest

def check_png(path: Path) -> None:
    if path.stat().st_size < 40000:
        raise ReviewBlocked("Missing or tiny image: " + path.name)
    with Image.open(path) as image:
        if image.format != "PNG" or image.width < 576 or image.height < 1024:
            raise ReviewBlocked("Wrong image format or size: " + path.name)
        if min(ImageStat.Stat(image.convert("RGB")).stddev) < 6:
            raise ReviewBlocked("Blank or near-blank image: " + path.name)

def audit(review_root: Path, approval_path: Path, *, require_motion: bool = False) -> dict:
    """Asset hashes must bind to a separately human-reviewed storyboard."""
    root = review_root.resolve()
    if not approval_path.is_file():
        raise ReviewBlocked("No human scene-by-scene visual review submitted")
    data = json.loads(approval_path.read_text(encoding="utf-8"))
    if data.get("schema") != 1 or data.get("episode") != "EB01":
        raise ReviewBlocked("Invalid episode review schema")
    if data.get("human_visual_review") is not True or data.get("approved") is not True:
        raise ReviewBlocked("Storyboard not personally accepted")
    if data.get("art_origin") == "synthetic_fixture":
        raise ReviewBlocked("Synthetic fixtures cannot pass release review")
    references = data.get("references", {})
    if set(references) != {"elizabeth", "kostya"}:
        raise ReviewBlocked("Require fixed canon references for BOTH characters")
    refs = {}
    for who, item in references.items():
        path = safely_resolve(root, item.get("file"))
        check_png(path)
        hash_ = sha256(path)
        if item.get("sha256") != hash_:
            raise ReviewBlocked("Character canon changed: " + who)
        if item.get("approved") is not True:
            raise ReviewBlocked("Character canon not approved: " + who)
        refs[who] = hash_
    entries = data.get("scenes", [])
    if not isinstance(entries, list) or len(entries) != len(SCENES):
        raise ReviewBlocked("Exactly seven locked story frames required")
    if [x.get("id") for x in entries] != list(SCENES):
        raise ReviewBlocked("Missing or rearranged frames")
    frame_hashes = {}
    for shot in entries:
        sid = shot["id"]
        path = safely_resolve(root, shot.get("file"))
        check_png(path)
        if set(shot.get("characters", [])) != CAST[sid]:
            raise ReviewBlocked("Wrong character cast in " + sid)
        if shot.get("sha256") != sha256(path):
            raise ReviewBlocked("Frame changed since review: " + sid)
        if not isinstance(shot.get("visual_checks"), dict):
            raise ReviewBlocked("No explicit scene inspection: " + sid)
        if any(shot["visual_checks"].get(key) is not True for key in CHECKS):
            raise ReviewBlocked("Failed frame quality/continuity check: " + sid)
        frame_hashes[sid] = sha256(path)
    if len(set(frame_hashes.values())) < 5:
        raise ReviewBlocked("Insufficient distinct compositions for seven-shot film")
    pilot = data.get("motion_pilot", {})
    if require_motion:
        if pilot.get("human_approved") is not True:
            raise ReviewBlocked("No human-approved *real* character animation test")
        if pilot.get("kind") != "image_to_video" or pilot.get("neural_actor_motion") is not True:
            raise ReviewBlocked("Camera pan/zoom is not character animation")
        path = safely_resolve(root, pilot.get("file"))
        if path.suffix.lower() != ".mp4" or pilot.get("sha256") != sha256(path):
            raise ReviewBlocked("Motion pilot missing or changed")
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,width,height", "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=25)
        info = json.loads(p.stdout)
        if not 1.0 <= float(info.get("format", {}).get("duration", 0)) <= 10:
            raise ReviewBlocked("Motion pilot duration invalid")
        if not any(x.get("codec_type") == "video" for x in info.get("streams", [])):
            raise ReviewBlocked("Motion pilot contains no video")
        for key in ("face_stable", "skeleton_stable", "physical_movement", "natural_motion"):
            if pilot.get(key) is not True:
                raise ReviewBlocked("Pilot animation quality failed: " + key)
    return {"status": "VISUAL_REVIEW_APPROVED",
            "references_sha256": refs, "seven_frames_sha256": frame_hashes,
            "motion_approved": require_motion, "gpu_jobs_launched": 0,
            "published": False}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--require-motion", action="store_true")
    options = p.parse_args()
    try:
        report = audit(options.assets, options.review, require_motion=options.require_motion)
    except (ReviewBlocked, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc),
                          "gpu_jobs_launched": 0, "published": False}, ensure_ascii=False))
        raise SystemExit(3)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
