"""Build an atomic, resumable three-part STORY GLITCH review package.

This consumes existing Work-authored manifests and actual media. It NEVER
writes stories, invents footage/voices, bypasses rights reviews, or publishes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

from studio.hybrid_master import check_manifest, produce
from studio.release_gate import check_video, sha256, cover_check

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,64}$")
CHECKSUM = re.compile(r"^[0-9a-f]{64}$")
PARTS = (1, 2, 3)


def inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Source path must be a relative string")
    base = root.resolve()
    target = (base / relative).resolve()
    if base not in target.parents or not target.is_file():
        raise ValueError(f"Missing or escaping source asset: {relative}")
    return target


def read_package_manifest(path: Path, *, master: bool = False) -> dict:
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("schema") != 1 or source.get("type") != "three_part_story":
        raise ValueError("Expected three_part_story schema=1")
    if not SAFE_ID.fullmatch(source.get("job_id", "")):
        raise ValueError("Missing safe job_id")
    story = source.get("story_title")
    if not isinstance(story, str) or not (4 <= len(story) <= 100):
        raise ValueError("Missing single story title for all three parts")
    refs = source.get("parts")
    if not isinstance(refs, list) or [part.get("part") if isinstance(part, dict) else None for part in refs] != list(PARTS):
        raise ValueError("Parts must be ordered 1/3, 2/3, 3/3")
    if master:
        if not (source.get("editorial_approved") is True
                and source.get("rights_approved") is True
                and source.get("final_human_review_approved") is True):
            raise ValueError("Master requires owner editorial, rights and final review approvals")
        if not CHECKSUM.fullmatch(source.get("work_script_sha256", "")):
            raise ValueError("Master requires the locked Work screenplay SHA256")
    base = path.resolve().parent
    planned = []
    seen_paths = set()
    shot_total = 0
    for part in refs:
        asset = inside(base, part.get("manifest"))
        cover = inside(base, part.get("cover"))
        if asset in seen_paths or cover in seen_paths:
            raise ValueError("Duplicate episode manifest/cover path")
        seen_paths.update((asset, cover))
        if asset.suffix.lower() != ".json" or cover.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Expected JSON manifest and image cover")
        manifest = json.loads(asset.read_text(encoding="utf-8"))
        # The existing hybrid editor enforces scene durations, media existence,
        # voice timings, source path boundaries and individual rights flags.
        plan = check_manifest(manifest, asset.parent, master=master)
        count = len(plan["shots"])
        expected = part.get("expected_shots", 8)
        if type(expected) is not int or expected < 1 or expected > 80 or count != expected:
            raise ValueError(f"Part {part['part']}: expected {expected} real shots, got {count}")
        shot_total += count
        if (plan["width"], plan["height"], plan["fps"]) != (1080, 1920, 30):
            raise ValueError("All episodes must be 1080x1920 at 30fps")
        voice_count = 0
        for ix, (raw, shot) in enumerate(zip(manifest["shots"], plan["shots"]), 1):
            if any(token in str(shot["path"]).lower() for token in ("/placeholder", "/dummy", "/fake_", "/demo_art")):
                raise ValueError(f"Part {part['part']} shot {ix}: procedural placeholder not allowed")
            media_hash = raw.get("media_sha256")
            if media_hash is not None and (not CHECKSUM.fullmatch(media_hash) or sha256(shot["path"]) != media_hash):
                raise ValueError(f"Part {part['part']} shot {ix}: image/clip SHA256 differs from manifest")
            if shot["kind"] == "image":
                with Image.open(shot["path"]) as image:
                    image.verify()
                with Image.open(shot["path"]) as image:
                    if min(image.size) < 540:
                        raise ValueError(f"Part {part['part']} shot {ix}: low-resolution image")
            if shot["voice"] is not None:
                voice_count += 1
                voice_hash = raw.get("voice_sha256")
                if voice_hash is not None and (not CHECKSUM.fullmatch(voice_hash)
                                               or sha256(shot["voice"]) != voice_hash):
                    raise ValueError("Narration SHA256 differs from manifest")
            if raw.get("requires_voice") is True and shot["voice"] is None:
                raise ValueError(f"Part {part['part']} shot {ix}: required WAV is missing")
            if master and not media_hash:
                raise ValueError("Master requires all source media SHA256 values")
            if master and shot["voice"] and not raw.get("voice_sha256"):
                raise ValueError("Master requires source voice SHA256")
        if voice_count == 0:
            raise ValueError(f"Part {part['part']}: real voiceover is missing")
        cover_info = cover_check(cover)
        planned.append({
            "part": part["part"], "manifest": asset, "cover": cover,
            "source_sha256": sha256(asset), "cover_sha256": cover_info["sha256"],
            "shots": count, "voice_tracks": voice_count,
            "duration_seconds": plan["duration"],
        })
    return {
        "job_id": source["job_id"], "story_title": story, "parts": planned,
        "shots": shot_total, "language": source.get("language", "ru"),
        "story_lock_sha256": source.get("work_script_sha256"),
        "master": bool(master), "source_manifest_sha256": sha256(path),
    }


def make_bundle(spec_file: Path, out_root: Path, *, master: bool = False,
                full_decode: bool = False) -> dict:
    spec = read_package_manifest(spec_file, master=master)
    out_root.mkdir(parents=True, exist_ok=True)
    root = out_root.resolve()
    target = root / (spec["job_id"] + ("_master" if master else "_review"))
    if target.exists():
        marker = target / "delivery.json"
        if marker.is_file():
            manifest = json.loads(marker.read_text(encoding="utf-8"))
            if manifest.get("source_manifest_sha256") == spec["source_manifest_sha256"]:
                # Never trust a stale marker without checking actual files.
                for row in manifest.get("parts", []):
                    check_video(target / row["mp4"], target / row["srt"],
                                target / row["cover"],
                                require_russian=spec["language"] == "ru",
                                full_decode=full_decode)
                if len(manifest.get("parts", [])) == 3:
                    return {"status": "cached_verified", "directory": str(target),
                            "story_title": spec["story_title"], "parts": 3}
        raise FileExistsError("Existing output is incomplete or has different source hash")
    with tempfile.TemporaryDirectory(prefix=".tw-staging-", dir=root) as temp_dir:
        temp = Path(temp_dir)
        delivery = {
            "schema": 1, "story_title": spec["story_title"], "job_id": spec["job_id"],
            "source_manifest_sha256": spec["source_manifest_sha256"],
            "work_script_sha256": spec["story_lock_sha256"],
            "parts": [], "shot_total": spec["shots"], "published": False,
            "quality": "technical_review_required",
            "owner_approval": "asserted_by_source" if master else "not_asserted",
        }
        for part in spec["parts"]:
            label = f"PART_{part['part']}_of_3"
            video = temp / (label + ".mp4")
            produce(part["manifest"], video, master=master)
            srt = video.with_suffix(".srt")
            cover = temp / (label + "_cover" + part["cover"].suffix.lower())
            cover.write_bytes(part["cover"].read_bytes())
            details = check_video(video, srt, cover,
                                  require_russian=spec["language"] == "ru",
                                  full_decode=full_decode)
            delivery["parts"].append({
                "part": part["part"], "mp4": video.name,
                "srt": srt.name, "cover": cover.name,
                "source_shots": part["shots"], "voice_tracks": part["voice_tracks"],
                "quality": details,
            })
        (temp / "delivery.json").write_text(json.dumps(delivery, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
        package = temp / (spec["job_id"] + "_review_bundle.zip")
        with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
            for f in sorted(temp.iterdir()):
                if f.is_file() and f != package:
                    archive.write(f, arcname=f.name)
        temp.rename(target)
    return {
        "status": "master_review_package" if master else "preview_review_package",
        "directory": str(target), "archive": str(target / package.name),
        "story_title": spec["story_title"], "parts": len(delivery["parts"]),
        "shots": spec["shots"], "video_master_quality_human_approved": False,
        "published": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Atomic 3-part production; no publishing, no invented media")
    p.add_argument("--package-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--master", action="store_true")
    p.add_argument("--full-decode", action="store_true")
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    result = (read_package_manifest(a.package_manifest, master=a.master)
              if a.preflight else
              make_bundle(a.package_manifest, a.output_dir, master=a.master,
                          full_decode=a.full_decode))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
