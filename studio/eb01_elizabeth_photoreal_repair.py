"""One-photo private Kaggle repair: original Kostya is never reuploaded.

The previous SaveKernel 400 run nested the original large skeleton PNG
in two base64 layers. This submitter performs source SHA validation but sends
ONLY a small string to the PRIVATE Elizabeth-only GPU worker. A read-only
post-process later combines proofs from two separate private Kaggle kernels.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path

from studio.eb01_canon_kaggle import (
    ARCHIVE as PARENT_ARCHIVE, MODEL as ORIGINAL_MODEL, verify as verify_canon
)
from studio.kaggle_api import ROOT
from studio.kaggle_identity import HANDLE

WORKER = "tools/kaggle_eb01_elizabeth_repair_entry.py"
MODEL = "RunDiffusion/Juggernaut-XL-v9"
ARCHIVE = "EB01_ELIZABETH_PRIVATE_ONLY.zip"
RECEIPT = "EB01_ELIZABETH_REPAIR_JOB.json"
CONTACT = "EB01_ELIZABETH_REVIEW_LOW_RES.jpg"
MAX_NOTEBOOK_SOURCE_BYTES = 128_000
PARENT_KERNEL = "wolf19loner92/eb01-v6-canon-adad762a9834"

def original_kostya(root: Path, nonce: str) -> tuple[bytes, str]:
    """Read-only verification on short-lived runner; NEVER serialize bytes to Kaggle."""
    source = verify_canon(root, nonce)
    if source.get("status") != "TWO_PRIVATE_REAL_CANDIDATES_REQUIRE_HUMAN_REVIEW":
        raise ValueError("Existing canonical skeleton source failed provenance check")
    files = list(Path(root).rglob(PARENT_ARCHIVE))
    if len(files) != 1:
        raise FileNotFoundError("Expected only one original private skeleton archive")
    with zipfile.ZipFile(files[0]) as z:
        original = z.read("canon/KOSTYA.png")
    sha = hashlib.sha256(original).hexdigest()
    if sha != source["images"]["KOSTYA"]["sha256"] or len(original) < 100000:
        raise ValueError("Private skeleton changed before photo repair")
    return original, sha

def prepare(username: str, nonce: str, previous_folder: Path,
            previous_nonce: str, destination: Path, root: Path = ROOT) -> dict:
    if not HANDLE.fullmatch(username) or not re.fullmatch("[A-Za-z0-9_-]{6,90}", nonce):
        raise ValueError("Invalid private Kaggle account or nonce")
    _, sha = original_kostya(previous_folder, previous_nonce)
    root = Path(root).resolve()
    worker = (root / WORKER).resolve()
    if root not in worker.parents or not worker.is_file():
        raise ValueError("Worker must be tracked in own repository")
    source = worker.read_text(encoding="utf-8")
    if source.count("__EB01_ELIZABETH_REPAIR_CONFIG_B64__") != 1:
        raise ValueError("Worker config marker mismatch; stop before spending quota")
    payload = {
        "schema": 2, "nonce": nonce, "model": MODEL,
        "source_revision": os.environ.get("GITHUB_SHA", "local-test-revision"),
        "parent_source_kernel": PARENT_KERNEL,
        "parent_KOSTYA_sha256": sha, "elizabeth_seed": 820261113,
        "reused_kostya": True, "max_new_images": 1,
        "allow_video": False, "private": True
    }
    # Absolutely no original PNG, no encoded nested PNG, no secret in source.
    assert "kostya_png_b64" not in payload
    serial = base64.b64encode(json.dumps(payload, sort_keys=True).encode()).decode()
    rendered = source.replace("__EB01_ELIZABETH_REPAIR_CONFIG_B64__", serial)
    bytes_ = rendered.encode("utf-8")
    if len(bytes_) > MAX_NOTEBOOK_SOURCE_BYTES:
        raise ValueError("Private Kaggle notebook payload size gate blocked request")
    if any(fragment in rendered for fragment in ("KAGGLE_API_TOKEN=", "hf_")):
        raise ValueError("Unsafe credential material in worker")

    slug = "eb01-v6-photoreal-" + hashlib.sha256(nonce.encode()).hexdigest()[:12]
    metadata = {
        "id": username + "/" + slug, "title": slug,
        "code_file": "kernel_main.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": []
    }
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "kernel_main.py").write_bytes(bytes_)
    (dest / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "status": "ELIZABETH_ONLY_SMALL_PRIVATE_JOB_READY",
        "kernel": metadata["id"], "source_model": MODEL,
        "original_KOSTYA_sha256": sha, "private_original_source_unchanged": True,
        "new_images": 1, "reused_good_existing_images": 1,
        "submitted_notebook_bytes": len(bytes_),
        "binary_image_embedded_in_notebook": False,
        "video_generated": False, "gpu_executed": False, "published": False
    }

def verify(path: Path, nonce: str, original_KOSTYA_sha256: str) -> dict:
    root = Path(path)
    receipts = list(root.rglob(RECEIPT))
    archives = list(root.rglob(ARCHIVE))
    previews = list(root.rglob(CONTACT))
    if len(receipts) != 1 or len(archives) != 1 or len(previews) != 1:
        raise FileNotFoundError("Expected exactly one Elizabeth-only private output set")
    data = json.loads(receipts[0].read_text(encoding="utf-8"))
    if not (
        data.get("schema") == 2 and data.get("success") is True
        and data.get("nonce") == nonce and data.get("model") == MODEL
        and data.get("parent_KOSTYA_sha256") == original_KOSTYA_sha256
        and data.get("parent_source_kernel") == PARENT_KERNEL
        and data.get("original_KOSTYA_untouched") is True
        and data.get("gpu_inference_performed") is True
        and data.get("image_identity_verified") is False
        and data.get("full_video_allowed") is False
        and data.get("video_generated") is False
        and data.get("published") is False
        and data.get("private") is True
    ):
        raise ValueError("Incomplete model, source, GPU or privacy provenance")
    sha_archive = hashlib.sha256(archives[0].read_bytes()).hexdigest()
    sha_preview = hashlib.sha256(previews[0].read_bytes()).hexdigest()
    if sha_archive != data.get("archive_sha256") or sha_preview != data.get("preview_sha256"):
        raise ValueError("Repair source/checksum mismatch")
    rows = data.get("generated_images", {})
    if set(rows) != {"ELIZABETH"}:
        raise ValueError("Worker must generate only one Elizabeth image, not Kostya")
    with zipfile.ZipFile(archives[0]) as z:
        if set(z.namelist()) != {"canon/ELIZABETH.png"}:
            raise ValueError("Original package contains unexpected files")
        original = z.read("canon/ELIZABETH.png")
    row = rows["ELIZABETH"]
    if (not original.startswith(b"\x89PNG\r\n\x1a\n")
            or len(original) != row.get("size")
            or len(original) < 120000
            or hashlib.sha256(original).hexdigest() != row.get("sha256")
            or row.get("width") != 704 or row.get("height") != 1216
            or row.get("model") != MODEL or row.get("art_reviewed") is not False):
        raise ValueError("Elizabeth original failed byte-identical provenance")
    return {
        "status": "PHOTOREAL_ELIZABETH_REPAIRED_KOSTYA_PRESERVED",
        "nonce": nonce, "source_model_KOSTYA": ORIGINAL_MODEL,
        "source_model_ELIZABETH": MODEL,
        "unchanged_KOSTYA_SHA256": original_KOSTYA_sha256,
        "ELIZABETH_sha256": row["sha256"],
        "visual_identity_approved": False, "human_review_required": True,
        "original_PNGs_on_GitHub": False, "video_generated": False,
        "published": False
    }

def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--username", required=True)
    a.add_argument("--nonce", required=True)
    a.add_argument("--source-output", type=Path, required=True)
    a.add_argument("--source-nonce", required=True)
    a.add_argument("--destination", type=Path, required=True)
    b = sub.add_parser("verify")
    b.add_argument("--path", type=Path, required=True)
    b.add_argument("--nonce", required=True)
    b.add_argument("--parent-kostya-sha256", required=True)
    x = p.parse_args(argv)
    result = (prepare(x.username, x.nonce, x.source_output, x.source_nonce, x.destination)
              if x.action == "prepare"
              else verify(x.path, x.nonce, x.parent_kostya_sha256))
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
