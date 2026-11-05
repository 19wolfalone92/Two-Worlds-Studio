"""Import ONE privately retrieved Kaggle ComfyUI PNG into review staging.

This closes the Kaggle -> Studio file handoff WITHOUT publishing or trusting
unreviewed art. No network, no model download, no script mutation.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

SHA = re.compile(r"^[0-9a-f]{64}$")
NONCE = re.compile(r"^[A-Za-z0-9_-]{3,100}$")
MAX_ZIP_BYTES = 30 * 1024 * 1024
MAX_PNG_BYTES = 25 * 1024 * 1024


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_archive(archive: Path, receipt: Path, *, nonce: str) -> dict:
    if not NONCE.fullmatch(nonce):
        raise ValueError("Expected safe Kaggle run nonce")
    if not archive.is_file() or archive.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError("Kaggle image bundle is absent or too large")
    if not receipt.is_file() or receipt.stat().st_size > 65536:
        raise ValueError("Kaggle receipt is absent or too large")
    status = json.loads(receipt.read_text(encoding="utf-8"))
    if status.get("success") is not True or status.get("nonce") != nonce:
        raise ValueError("Kaggle receipt is unsuccessful or from another run")
    with zipfile.ZipFile(archive, "r") as z:
        entries = z.infolist()
        names = [x.filename for x in entries]
        if len(entries) != 2 or set(names) != {"render.png", "render_manifest.json"}:
            raise ValueError("Expected exactly one real PNG and its manifest")
        if any(x.file_size > MAX_PNG_BYTES or x.is_dir()
               or (x.external_attr >> 16) & 0o170000 == 0o120000 for x in entries):
            raise ValueError("Unsafe Kaggle archive entry")
        png = z.read("render.png")
        manifest = json.loads(z.read("render_manifest.json"))
    if len(png) < 10000 or not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Missing real rendered PNG")
    with Image.open(io.BytesIO(png)) as image:
        image.verify()
    with Image.open(io.BytesIO(png)) as image:
        resolution = image.size
        if image.format != "PNG":
            raise ValueError("Decoded image is not PNG")
    if resolution != (640, 1152):
        raise ValueError("Image does not have the expected GPU proof resolution")
    if (manifest.get("png_sha256") != sha(png)
            or manifest.get("comfy_prompt_id") != status.get("comfy_prompt_id")
            or manifest.get("review_status") != "unreviewed"
            or not isinstance(manifest.get("comfy_prompt_id"), str)):
        raise ValueError("Image provenance/receipt/quality state does not match")
    if manifest.get("no_publication") is not True:
        raise ValueError("Unreviewed generated art cannot be auto-published")
    return {
        "run_nonce": nonce, "png_sha256": sha(png),
        "width": resolution[0], "height": resolution[1],
        "comfy_prompt_id": manifest["comfy_prompt_id"],
        "scene": manifest.get("scene"), "source_sha": manifest.get("source_sha"),
        "art_review_required": True, "rights_review_required": True,
        "master_ready": False, "published": False,
        "bytes": png,
    }


def stage(archive: Path, receipt: Path, *, nonce: str, destination: Path) -> dict:
    findings = inspect_archive(archive, receipt, nonce=nonce)
    dest = destination.resolve()
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite staged art: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".comfy-art-", dir=dest.parent) as td:
        current = Path(td)
        (current / "render.png").write_bytes(findings["bytes"])
        info = {k: v for k, v in findings.items() if k != "bytes"}
        (current / "art_review.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        current.rename(dest)
    return {"status": "staged_unreviewed", "directory": str(dest),
            "image": str(dest / "render.png"), "sha256": info["png_sha256"],
            "published": False}


def main():
    parser = argparse.ArgumentParser(description="Import private Kaggle art locally (never auto-approves)")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.archive, args.receipt, nonce=args.nonce,
                           destination=args.dest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
