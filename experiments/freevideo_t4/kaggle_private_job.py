"""Build a PRIVATE Kaggle T4 kernel for an owner's image — CPU only.

No Kaggle API usage in this builder, no GitHub writes, no account creation.
The original photo is embedded ONLY into the generated private Kaggle script,
never into tracked source, public GitHub artifacts or workflow logs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

from PIL import Image

try:
    from .hybrid_engine import ROOT, private_path, sha256
except ImportError:
    from hybrid_engine import ROOT, private_path, sha256

ACCOUNT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,29}\Z")
NONCE = re.compile(r"[a-z0-9][a-z0-9_-]{4,59}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
TEMPLATE = ROOT / "experiments/freevideo_t4/kaggle_worker_template.py"
MARKERS = ("__FREEVIDEO_JOB_CONFIG_B64__", "__FREEVIDEO_PRIVATE_IMAGE_B64__")


class KaggleBlocked(ValueError):
    """Stop before uploading a photo or spending Kaggle quota."""


def prepare_private_kernel(username: str, nonce: str, source: str | Path,
                           prompt: str, destination: str | Path, *,
                           revision: str, seed: int = 42,
                           accept_model_license: bool = False) -> dict:
    if accept_model_license is not True:
        raise KaggleBlocked("Explicit official model licence review is required")
    if not isinstance(username, str) or not ACCOUNT.fullmatch(username):
        raise KaggleBlocked("Invalid Kaggle username")
    if not isinstance(nonce, str) or not NONCE.fullmatch(nonce):
        raise KaggleBlocked("Invalid private job nonce")
    if not isinstance(revision, str) or not SHA.fullmatch(revision):
        raise KaggleBlocked("Pin an exact 40-hex Freevideo experimental Git SHA")
    if not isinstance(prompt, str) or not 12 <= len(prompt.strip()) <= 1800:
        raise KaggleBlocked("Provide a 12-1800-character natural motion prompt")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise KaggleBlocked("Invalid model seed")
    image = private_path(source)
    if image.suffix.lower() != ".png":
        raise KaggleBlocked("Use an already decoded private PNG as Kaggle reference")
    if image.stat().st_size > 8 * 1024 * 1024:
        raise KaggleBlocked("For the private kernel embed, source PNG must be <=8 MiB")
    try:
        with Image.open(image) as pic:
            pic.verify()
    except OSError as exc:
        raise KaggleBlocked("Private PNG cannot be decoded") from exc
    target = Path(destination).expanduser().absolute()
    if (target.is_symlink() or target.exists() or target.resolve().is_relative_to(ROOT)
            or target.is_relative_to(ROOT)):
        raise KaggleBlocked("Destination must be a NEW private directory outside public GitHub checkout")
    worker = TEMPLATE.read_text(encoding="utf-8")
    if any(worker.count(m) != 1 for m in MARKERS):
        raise KaggleBlocked("Unexpected Kaggle worker template marker count")
    payload = {
        "schema": 1, "model": "wan22-5b", "nonce": nonce,
        "source_revision": revision, "reference_sha256": sha256(image),
        "license_reviewed": True, "prompt": prompt.strip(), "seed": seed,
        "publications_authorized": False}
    rendered = worker.replace(
        MARKERS[0], base64.b64encode(json.dumps(payload, sort_keys=True).encode()).decode()
    ).replace(MARKERS[1], base64.b64encode(image.read_bytes()).decode())
    if any(marker in rendered for marker in MARKERS):
        raise KaggleBlocked("Private Kaggle template marker not expanded")
    slug = "freevideo-hybrid-" + hashlib.sha256(nonce.encode()).hexdigest()[:12]
    metadata = {
        "id": username + "/" + slug, "title": slug,
        "code_file": "kernel_main.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": []}
    target.mkdir(parents=True, mode=0o700)
    try:
        script = target / "kernel_main.py"
        with script.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
        os.chmod(script, 0o600)
        with (target / "kernel-metadata.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
        os.chmod(target / "kernel-metadata.json", 0o600)
    except BaseException:
        for item in target.iterdir():
            item.unlink(missing_ok=True)
        target.rmdir()
        raise
    return {
        "status": "PRIVATE_KAGGLE_PACKAGE_PREPARED_NO_GPU",
        "kernel": metadata["id"], "nonce": nonce,
        "source_sha256": payload["reference_sha256"], "git_sha": revision,
        "script_bytes": len(rendered.encode()), "source_private": True,
        "model": "wan22-5b", "gpu_executed": False, "published": False,
        "warning": "Private kernel contains user's source photo; never publish or add to GitHub."}


def verify_private_result(folder: str | Path, nonce: str) -> dict:
    """Validate returned Kaggle archive without trusting declared success."""
    location = Path(folder).expanduser().absolute()
    if not NONCE.fullmatch(nonce):
        raise KaggleBlocked("Invalid nonce")
    manifest = private_path(location / "freevideo_result.json")
    archive = private_path(location / "FREEVIDEO_PRIVATE_PREVIEW.zip")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("nonce") != nonce
            or record.get("status") != "REAL_GPU_PREVIEW_REVIEW_REQUIRED"
            or record.get("human_approved") is not False
            or record.get("published") is not False
            or record.get("visual_quality") != "NOT_VERIFIED"):
        raise KaggleBlocked("No matching successful GPU receipt")
    if not 8000 <= archive.stat().st_size <= 300 * 1024 * 1024:
        raise KaggleBlocked("Private preview archive size invalid")
    if sha256(archive) != record.get("archive_sha256"):
        raise KaggleBlocked("Kaggle ZIP SHA mismatch")
    wanted = {"preview.mp4", "preview.receipt.json", "preview.freevideo.json",
              "preview.hybrid.json", "job.json"}
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        names = [m.filename for m in members]
        if (len(names) != len(set(names)) or not wanted.issubset(names)
                or set(names) - wanted - {"timings.jsonl"}
                or any(m.is_dir() or m.file_size > 300 * 1024 * 1024
                       or (m.external_attr >> 16) & 0o170000 == 0o120000
                       for m in members)):
            raise KaggleBlocked("Unsafe/missing private preview ZIP members")
        raw = z.read("preview.mp4")
        studio = json.loads(z.read("preview.receipt.json"))
        hybrid = json.loads(z.read("preview.hybrid.json"))
        freevideo = json.loads(z.read("preview.freevideo.json"))
        digest = hashlib.sha256(raw).hexdigest()
        if (digest != record.get("clip_sha256")
                or digest != studio.get("clip_sha256")
                or digest != hybrid.get("clip_sha256")
                or digest != freevideo.get("clip_sha256")
                or hybrid.get("stage") != "preview"
                or studio.get("inference_performed") is not True):
            raise KaggleBlocked("Native MP4/provenance mismatch")
    # Do not claim artistic quality or actual decoded frame validation without
    # decoding the returned clip. Hash alone is not proof of a valid MP4.
    return {"status": "HASHES_MATCH_GPU_RECEIPTS_DECODE_NOT_VERIFIED",
            "clip_sha256": record["clip_sha256"], "nonce": nonce,
            "actual_decoding": "NOT_TESTED", "visual_quality": "NOT_VERIFIED",
            "human_approved": False, "published": False}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build/verify only a PRIVATE Kaggle Freevideo image-to-video job")
    cmd = p.add_subparsers(dest="cmd", required=True)
    a = cmd.add_parser("prepare")
    for key in ("username", "nonce", "source", "prompt", "destination", "revision"):
        a.add_argument("--" + key.replace("_", "-"), required=True)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--accept-model-license", action="store_true")
    b = cmd.add_parser("verify")
    b.add_argument("--path", required=True)
    b.add_argument("--nonce", required=True)
    v = p.parse_args(argv)
    try:
        if v.cmd == "prepare":
            result = prepare_private_kernel(v.username, v.nonce, v.source,
                                            v.prompt, v.destination,
                                            revision=v.revision, seed=v.seed,
                                            accept_model_license=v.accept_model_license)
        else:
            result = verify_private_result(v.path, v.nonce)
        code = 0
    except (KaggleBlocked, OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        result, code = {"status": "BLOCKED", "reason": str(exc),
                        "gpu_success": False, "published": False}, 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
