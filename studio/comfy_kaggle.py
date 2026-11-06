"""Prepare a private, one-shot Kaggle GPU job running ComfyUI locally.

This is an isolated adapter: does not replace the working FLUX/Kokoro pipeline,
send credentials to Kaggle, publish video, or approve generated artwork.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import struct
import zipfile
from pathlib import Path

from studio.kaggle_api import ROOT, USERNAME, package_source, poll_kernel

SLUG = "two-worlds-comfy-sdxl"
RECEIPT = "two_worlds_job_receipt.json"
PACKAGE = "TWO_WORLDS_ARTIFACTS.zip"
EXTRA_FILES = ("templates/comfyui_sdxl_api.json", "episodes/E01_THE_REFLECTION.json")
CONFIG_MARKER = "__TW_COMFY_CONFIG__"
SOURCE_MARKER = "__TW_COMFY_SOURCE__"


def reviewed_source(root: Path = ROOT) -> bytes:
    """Bundle only the existing vetted studio source and public test fixture."""
    base = package_source(root)
    memory = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(base)) as old, zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as new:
        for name in old.namelist():
            new.writestr(name, old.read(name))
        for name in EXTRA_FILES:
            path = (root / name).resolve()
            if not path.is_file() or root.resolve() not in path.parents:
                raise FileNotFoundError(name)
            new.write(path, arcname=name)
    if memory.tell() > 2_000_000:
        raise ValueError("ComfyUI source package exceeds audited 2MB limit")
    return memory.getvalue()


def build_job(username: str, nonce: str, *, output_dir: Path,
              source_root: Path = ROOT, git_sha: str | None = None) -> dict:
    if not USERNAME.fullmatch(username):
        raise ValueError("Invalid Kaggle username")
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,90}", nonce):
        raise ValueError("Invalid run nonce")
    sha = git_sha if git_sha is not None else os.environ.get("GITHUB_SHA", "local-test")
    if not isinstance(sha, str) or not re.fullmatch(r"[A-Za-z0-9_-]{3,80}", sha):
        raise ValueError("Invalid revision")
    metadata = {
        "id": f"{username}/{SLUG}",
        "title": "Two Worlds Comfy SDXL",
        "code_file": "kernel_main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    source = reviewed_source(source_root)
    payload = {
        "schema": 1, "nonce": nonce, "source_sha": sha,
        "checkpoint_repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "checkpoint_file": "sd_xl_base_1.0.safetensors",
        "episode_fixture": "episodes/E01_THE_REFLECTION.json",
        "scene": "S01", "width": 640, "height": 1152,
    }
    template = (source_root / "tools/comfy_kaggle_entry.py").read_text(encoding="utf-8")
    if template.count(CONFIG_MARKER) != 1 or template.count(SOURCE_MARKER) != 1:
        raise ValueError("Invalid ComfyUI worker template")
    script = template.replace(
        CONFIG_MARKER, base64.b64encode(json.dumps(payload, sort_keys=True).encode()).decode()
    ).replace(SOURCE_MARKER, base64.b64encode(source).decode())
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "kernel_main.py").write_text(script, encoding="utf-8")
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "kernel": metadata["id"], "nonce": nonce,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_bytes": len(source), "credentials_embedded": False,
        "model": payload["checkpoint_repo"], "scene": payload["scene"],
    }


def verify(path: Path, nonce: str) -> dict:
    """Reject stale output, simulated images, wrong format or missing ComfyUI ID."""
    receipts = list(path.rglob(RECEIPT))
    archives = list(path.rglob(PACKAGE))
    if len(receipts) != 1 or len(archives) != 1:
        raise FileNotFoundError("Expected one receipt and one artifact archive")
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    if receipt.get("nonce") != nonce or receipt.get("success") is not True:
        raise ValueError("Missing successful receipt for this exact run nonce")
    if archives[0].stat().st_size > 100 * 1024 * 1024:
        raise ValueError("Kaggle archive exceeds 100MB proof limit")
    with zipfile.ZipFile(archives[0]) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(entries) != 2 or set(names) != {"render.png", "render_manifest.json"}:
            raise ValueError("Expected exactly the PNG and its manifest")
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("Unsafe archive entry names")
        if any(entry.file_size > 30 * 1024 * 1024 or entry.is_dir()
               or (entry.external_attr >> 16) & 0o170000 == 0o120000 for entry in entries):
            raise ValueError("Unsafe archive payload")
        png = archive.read("render.png")
        manifest = json.loads(archive.read("render_manifest.json"))
    if len(png) < 10_000 or png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("No real PNG output")
    width, height = struct.unpack(">II", png[16:24])
    if (width, height) != (640, 1152):
        raise ValueError("Unexpected dimensions")
    from studio.art_image_qc import InvalidArtwork, inspect_image_bytes
    try:
        image_qc = inspect_image_bytes(png, expected_size=(640, 1152))
    except InvalidArtwork:
        raise ValueError("ComfyUI PNG failed full pixel decoding") from None
    if manifest.get("comfy_prompt_id") != receipt.get("comfy_prompt_id") or not manifest.get("comfy_prompt_id"):
        raise ValueError("Missing matching ComfyUI prompt ID")
    if manifest.get("png_sha256") != hashlib.sha256(png).hexdigest():
        raise ValueError("PNG checksum mismatch")
    if manifest.get("review_status") != "unreviewed":
        raise ValueError("Artwork must remain unreviewed")
    return {
        "success": True, "nonce": nonce, "width": width, "height": height,
        "comfy_prompt_id": manifest["comfy_prompt_id"],
        "model": manifest.get("checkpoint_file"),
        "archive_sha256": hashlib.sha256(archives[0].read_bytes()).hexdigest(),
        "art_approved": False, "image_quality": image_qc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--username", required=True)
    prepare.add_argument("--nonce", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--nonce", required=True)
    check.add_argument("--output-dir", type=Path, required=True)
    poll = commands.add_parser("poll")
    poll.add_argument("--kernel", required=True)
    poll.add_argument("--timeout", type=int, default=3300)
    args = parser.parse_args()
    if args.command == "prepare":
        result = build_job(args.username, args.nonce, output_dir=args.output_dir)
    elif args.command == "verify":
        result = verify(args.output_dir, args.nonce)
    else:
        result = {"status": poll_kernel(args.kernel, timeout_seconds=args.timeout,
                                        interval_seconds=180)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
