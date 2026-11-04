"""Reconcile actual PRIVATE Kaggle media archives with the original six Work stories.

Fail closed: immutable original scripts; verified ZIP manifests; no synthetic
assets, no public upload, no mutation of the source story kit. The output queue
is published atomically only after all chosen archives were verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from .kaggle_shorts_worker import _safe_extract
from .autonomous_batch import plan

STORIES = tuple(str(n) for n in range(13, 19))
PHASES = ("image", "voice")
MAX_ZIP_BYTES = 750 * 1024 * 1024
MAX_UNPACKED_BYTES = 1600 * 1024 * 1024
SHA = re.compile(r"^[0-9a-f]{64}$")
IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


class ReconcileBlocked(ValueError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def project(source: Path, story: str) -> tuple[Path, dict]:
    folder = source / "production_queue" / story
    if not folder.is_dir() or folder.is_symlink():
        raise ReconcileBlocked("Original private story not found")
    doc_path = folder / "project.json"
    script = folder / "story.txt"
    if not doc_path.is_file() or not script.is_file() or script.is_symlink():
        raise ReconcileBlocked("Original locked story files missing")
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    if (doc.get("id") != f"work_{story}" or doc.get("schema") != 1
            or doc.get("language") != "ru" or len(doc.get("scenes", [])) != 8
            or doc.get("script_file") != "story.txt"
            or doc.get("script_sha256") != digest(script)):
        raise ReconcileBlocked("Original story and SHA256 disagree")
    return folder, doc


def _members(z: zipfile.ZipFile, story: str) -> dict[str, zipfile.ZipInfo]:
    result = {}
    total = 0
    for entry in z.infolist():
        name = entry.filename
        parts = Path(name).parts
        if (Path(name).is_absolute() or ".." in parts or name.startswith("./")
                or (entry.external_attr >> 16) & 0o170000 == 0o120000
                or name in result):
            raise ReconcileBlocked("Unsafe or duplicate private archive entry")
        total += entry.file_size
        if total > MAX_UNPACKED_BYTES or entry.file_size > MAX_ZIP_BYTES:
            raise ReconcileBlocked("Private recovery archive exceeds bound")
        if entry.is_dir():
            continue
        if name != "run_manifest.json" and parts[:2] != ("production_queue", story):
            raise ReconcileBlocked("Unexpected story found in private recovery archive")
        result[name] = entry
    if len(result) > 500:
        raise ReconcileBlocked("Too many private archive entries")
    return result


def import_phase(folder: Path, story: str, archive: Path, phase: str) -> dict:
    if phase not in PHASES or not archive.is_file() or archive.stat().st_size > MAX_ZIP_BYTES:
        raise ReconcileBlocked("Missing or oversized phase archive")
    original = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(archive) as z:
        members = _members(z, story)
        if "run_manifest.json" not in members:
            raise ReconcileBlocked("Missing private source manifest")
        receipt = json.loads(z.read("run_manifest.json"))
        if (receipt.get("story") != story or receipt.get("script_sha256") != original["script_sha256"]
                or not isinstance(receipt.get("generated_media"), list)):
            raise ReconcileBlocked("Archive provenance and original script do not match")
        expected = {}
        for item in receipt["generated_media"]:
            name, sha, size = item.get("path"), item.get("sha256"), item.get("bytes")
            if (not isinstance(name, str) or name not in members or not isinstance(sha, str)
                    or not SHA.fullmatch(sha) or type(size) is not int
                    or size != members[name].file_size or name in expected):
                raise ReconcileBlocked("Invalid archive SHA256 inventory")
            expected[name] = sha
        if set(expected) != set(members) - {"run_manifest.json"}:
            raise ReconcileBlocked("Unlisted private archive bytes")
        archived_doc_name = f"production_queue/{story}/project.json"
        if archived_doc_name not in expected:
            raise ReconcileBlocked("Missing original project metadata in recovery")
        stage_doc = json.loads(z.read(archived_doc_name))
        if (stage_doc.get("id") != original["id"]
                or stage_doc.get("script_sha256") != original["script_sha256"]
                or len(stage_doc.get("scenes", [])) != len(original["scenes"])):
            raise ReconcileBlocked("Wrong immutable story in phase archive")
        scene_ids = [s["id"] for s in original["scenes"]]
        if sorted(s.get("id") for s in stage_doc["scenes"]) != sorted(scene_ids):
            raise ReconcileBlocked("Scene indices changed")
        prefix = f"production_queue/{story}/"
        copied = 0
        for name, info in members.items():
            if name == "run_manifest.json":
                continue
            # Verify every archived payload, even metadata that isn't imported.
            with z.open(info) as input:
                h = hashlib.sha256()
                chunks = []
                while part := input.read(1024 * 1024):
                    h.update(part)
                    # Large private assets are streamed to disk in the second pass.
                if h.hexdigest() != expected[name]:
                    raise ReconcileBlocked("Private archive checksum mismatch")
        for name, info in members.items():
            if name == "run_manifest.json":
                continue
            rel = name[len(prefix):]
            path = Path(rel)
            if phase == "image":
                selected = (len(path.parts) >= 2 and path.parts[0] == "images"
                            and path.suffix.lower() in IMAGES)
            else:
                selected = (len(path.parts) >= 2 and path.parts[0] == "voices"
                            and path.suffix.lower() in AUDIO | {".json"})
            if not selected:
                continue
            target = (folder / path).resolve()
            if folder.resolve() not in target.parents or target.is_symlink():
                raise ReconcileBlocked("Media would escape private source folder")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != expected[name]:
                    raise ReconcileBlocked("Conflicting generated assets; choose one reviewed take")
                continue
            # Atomic individual file copy; abandoned build directory is private.
            temporary = target.with_name(target.name + ".partial")
            if temporary.exists():
                raise ReconcileBlocked("Previous incomplete temporary asset")
            with z.open(info) as incoming, temporary.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            os.replace(temporary, target)
            copied += 1
        updated = {s["id"]: s for s in stage_doc["scenes"]}
        for scene in original["scenes"]:
            incoming = updated[scene["id"]]
            if phase == "image":
                image_name = incoming.get("image")
                if not isinstance(image_name, str) or not image_name.startswith("images/"):
                    raise ReconcileBlocked("Invalid real image path")
                if not (folder / image_name).is_file():
                    continue
                scene["image"] = image_name
            else:
                voice_name = incoming.get("voice")
                if not isinstance(voice_name, str) or not voice_name.startswith("voices/"):
                    raise ReconcileBlocked("Invalid real voice path")
                if not (folder / voice_name).is_file():
                    continue
                scene["voice"] = voice_name
                scene["subtitles"] = incoming.get("subtitles", [])
                if incoming.get("subtitle_alignment"):
                    scene["subtitle_alignment"] = incoming["subtitle_alignment"]
        temp = folder / ".project-reconcile.json"
        temp.write_text(json.dumps(original, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temp, folder / "project.json")
    return {"phase": phase, "verified_archive_sha256": digest(archive),
            "copied_media_files": copied, "source_script_sha256": original["script_sha256"]}


def reconcile(source_path: Path, media_root: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise ReconcileBlocked("Refuse to overwrite existing private output")
    if not media_root.is_dir() or media_root.is_symlink():
        raise ReconcileBlocked("Private media directory not found")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".private-batch-join-", dir=output.parent) as tmp:
        stage = Path(tmp) / "queue"
        stage.mkdir()
        with tempfile.TemporaryDirectory(prefix=".private-batch-source-", dir=output.parent) as original_tmp:
            if source_path.is_dir():
                root = source_path.resolve()
            elif source_path.is_file() and source_path.suffix == ".zip":
                if source_path.stat().st_size > 180 * 1024 * 1024:
                    raise ReconcileBlocked("Oversized locked source ZIP")
                root = Path(original_tmp)
                with zipfile.ZipFile(source_path) as z:
                    _safe_extract(z, root)
            else:
                raise ReconcileBlocked("Original Work Batch03 source is absent")
            manifest = []
            for story in STORIES:
                origin, _ = project(root, story)
                destination = stage / story
                shutil.copytree(origin, destination, symlinks=False)
                phases = []
                for phase in PHASES:
                    name = f"TwoWorlds_PRIVATE_{story}_{phase}.zip"
                    archive = media_root / name
                    if archive.is_symlink():
                        raise ReconcileBlocked("Symlinked private media input")
                    if archive.is_file():
                        phases.append(import_phase(destination, story, archive, phase))
                manifest.append({"story": story, "phases": phases})
        # Keep the queue private even when only some images/voices exist.
        os.rename(stage, output)
    result = plan(output, output.parent / "unpublished_exports")
    return {"status": "PRIVATE_QUEUE_IMPORTED", "source_stories": 6,
            "imported": manifest, "production": result,
            "paid_calls": False, "published": False}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Merge real private image+voice ZIPs into locked Batch03")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--media", type=Path, required=True)
    p.add_argument("--queue", type=Path, required=True)
    args = p.parse_args(argv)
    doc = reconcile(args.source, args.media, args.queue)
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
