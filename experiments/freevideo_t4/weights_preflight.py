"""Offline integrity gate for installed Wan2.2 ComfyUI weight files.

No model load/download. Hashes are compared to OPERATOR-SUPPLIED manifests:
this is not an independently authenticated publisher signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
MODEL = "wan22-5b"
REQUIRED = frozenset((
    "wan2.2_ti2v_5B_fp16.safetensors",
    "wan2.2_vae.safetensors",
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
))
MAX_MANIFEST_BYTES = 32768

class IntegrityBlocked(ValueError):
    """Private weight file or manifest fails integrity gating."""

def _private_file(raw: str | Path, *, repo_root: Path = ROOT) -> Path:
    lexical = Path(raw).expanduser().absolute()
    if lexical.is_relative_to(repo_root) or lexical.is_symlink():
        raise IntegrityBlocked("Do not read public-repo files or symlinked assets")
    path = lexical.resolve(strict=True)
    if path.is_relative_to(repo_root) or not path.is_file():
        raise IntegrityBlocked("Expected a private regular file")
    return path

def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityBlocked("Manifest has duplicate JSON keys")
        result[key] = value
    return result

def verify_weights_manifest(manifest_path: str | Path) -> dict:
    path = _private_file(manifest_path)
    if not 128 <= path.stat().st_size <= MAX_MANIFEST_BYTES:
        raise IntegrityBlocked("Manifest size outside limit")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityBlocked("Manifest JSON cannot be parsed") from exc
    if not isinstance(doc, dict) or doc.get("schema") != 1 or doc.get("model") != MODEL:
        raise IntegrityBlocked("Expected schema 1 Wan2.2 TI2V-5B manifest")
    if doc.get("license_reviewed") is not True or not isinstance(doc.get("source_repository"), str):
        raise IntegrityBlocked("Explicit per-install model licence/source review is required")
    source = doc["source_repository"]
    if not source.startswith("https://") or len(source) > 250:
        raise IntegrityBlocked("Model source must be an explicit HTTPS repository")
    files = doc.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED:
        raise IntegrityBlocked("Manifest must list exactly the three installed bridge weights")
    matches = []
    for filename in sorted(REQUIRED):
        entry = files[filename]
        if not isinstance(entry, dict):
            raise IntegrityBlocked("Invalid per-weight manifest entry")
        expected = entry.get("sha256")
        raw_path = entry.get("path")
        if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
            raise IntegrityBlocked("Expected SHA-256 must be a lowercase 64-hex digest")
        if not isinstance(raw_path, str) or not raw_path:
            raise IntegrityBlocked("Missing weight path")
        weight = _private_file(raw_path)
        if weight.name != filename or not weight.stat().st_size:
            raise IntegrityBlocked("Installed weight basename/size mismatch")
        hasher = hashlib.sha256()
        with weight.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != expected:
            raise IntegrityBlocked(f"Weight digest mismatch: {filename}")
        matches.append({"name": filename, "sha256": expected, "bytes": weight.stat().st_size})
    return {
        "schema": 1, "status": "LOCAL_HASHES_MATCH_OPERATOR_MANIFEST",
        "model": MODEL, "components": matches, "license_reviewed": True,
        "publisher_signature_verified": False, "real_gpu_tested": False,
        "limitation": "The supplied expected hashes and licence assertion are not independently authenticated.",
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare private model files to a private manifest")
    parser.add_argument("manifest")
    args = parser.parse_args(argv)
    try:
        doc, rc = verify_weights_manifest(args.manifest), 0
    except (IntegrityBlocked, OSError) as exc:
        doc, rc = {"status": "BLOCKED", "reason": str(exc), "real_gpu_tested": False}, 2
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
