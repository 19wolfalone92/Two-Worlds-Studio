"""Exactly one private Kaggle Wan2.2 video-GPU proof; no user media in public Git.

Packaging is CPU-only. A separately authorized workflow pushes this source
to a private Kaggle kernel and runs the original ComfyUI model exactly once.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

from studio.kaggle_api import ROOT
from studio.kaggle_identity import HANDLE
from studio.kaggle_api_limits import output_once
from studio.release_gate import sha256

WORKER = "tools/kaggle_wan22_video_entry.py"
FILESET = ("studio/__init__.py", "studio/comfy_neural.py", "studio/video_ai_plan.py",
           "studio/model_install.py", "studio/release_gate.py")
MODELS = {"wan22-5b"}
VIDEO_ARCHIVE = "TWO_WORLDS_ARTIFACTS.zip"
RECEIPT = "two_worlds_job_receipt.json"
FRAME_COUNT = 17
WIDTH, HEIGHT = 448, 800
STEPS = 20


def _bundle(root: Path) -> bytes:
    root = root.resolve()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in FILESET:
            full = (root / relative).resolve()
            if (not full.is_file() or root not in full.parents or full.stat().st_size > 150000
                    or full.is_symlink()):
                raise ValueError("Unreviewed or absent source module: " + relative)
            archive.write(full, relative)
    if stream.tell() > 600_000:
        raise ValueError("Worker source bundle unexpectedly large")
    return stream.getvalue()


def prepare(username: str, nonce: str, destination: Path, *, root: Path = ROOT,
            revision: str | None = None) -> dict:
    if not HANDLE.fullmatch(username):
        raise ValueError("Invalid Kaggle account name")
    if not re.fullmatch(r"[A-Za-z0-9_-]{5,90}", nonce):
        raise ValueError("Invalid run nonce")
    sha = revision if revision is not None else os.environ.get("GITHUB_SHA", "local-code-test")
    if not isinstance(sha, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{8,80}", sha):
        raise ValueError("Invalid source revision")
    root = root.resolve()
    worker = (root / WORKER).resolve()
    if not worker.is_file() or root not in worker.parents:
        raise FileNotFoundError(WORKER)
    source = worker.read_text(encoding="utf-8")
    if source.count("__TW_VIDEO_CONFIG__") != 1 or source.count("__TW_VIDEO_BUNDLE__") != 1:
        raise ValueError("Worker template has unexpected placeholders")
    blob = _bundle(root)
    slug = "two-worlds-wan5b-" + hashlib.sha256(nonce.encode()).hexdigest()[:12]
    metadata = {
        "id": username + "/" + slug,
        "title": slug,  # Kaggle enforces title->slug matching; otherwise it saves under the title slug.
        "code_file": "kernel_main.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true",
        "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    payload = {
        "schema": 1, "nonce": nonce, "source_revision": sha, "model": "wan22-5b",
        "width": WIDTH, "height": HEIGHT, "frames": FRAME_COUNT, "steps": STEPS,
        "seed": 20261111, "prompt": (
            "An illustrated woman in a dark jacket gently turns her head and shoulders "
            "toward the viewer, takes one natural breath and slowly raises her right hand. "
            "Coherent body motion, soft indoor lighting, stable camera, consistent anatomy."
        ),
    }
    rendered = source.replace(
        "__TW_VIDEO_CONFIG__", base64.b64encode(json.dumps(payload, sort_keys=True).encode()).decode()
    ).replace("__TW_VIDEO_BUNDLE__", base64.b64encode(blob).decode())
    if any(k in rendered for k in ("__TW_VIDEO_CONFIG__", "__TW_VIDEO_BUNDLE__")):
        raise ValueError("Unexpanded private worker template marker")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "kernel_main.py").write_text(rendered, encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "kernel": metadata["id"], "nonce": nonce, "source_sha256": hashlib.sha256(blob).hexdigest(),
        "generated_script_bytes": len(rendered), "model": "wan22-5b",
        "frames": FRAME_COUNT, "size": [WIDTH, HEIGHT],
        "gpu_executed": False, "credentials_included": False, "published": False,
    }


def verify(path: Path, nonce: str) -> dict:
    root = Path(path)
    receipts = list(root.rglob(RECEIPT))
    archives = list(root.rglob(VIDEO_ARCHIVE))
    if len(receipts) != 1 or len(archives) != 1:
        raise FileNotFoundError("Missing unique Kaggle output receipt and video archive")
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    if (receipt.get("nonce") != nonce or receipt.get("success") is not True
            or receipt.get("model") != "wan22-5b"
            or receipt.get("inference_performed") is not True
            or receipt.get("video", {}).get("frames") != FRAME_COUNT
            or receipt.get("video", {}).get("width") != WIDTH
            or receipt.get("video", {}).get("height") != HEIGHT):
        raise ValueError("No verified full-frame Wan2.2 inference for this run nonce")
    archive = archives[0]
    if not 8000 < archive.stat().st_size < 160_000_000:
        raise ValueError("Unrealistic Kaggle video archive size")
    # A matching MP4 hash alone does not bind the ZIP metadata and QC notes.
    # The worker produces an overall archive SHA. Verify it before unpacking.
    expected_archive_sha = receipt.get("archive_sha256")
    if not isinstance(expected_archive_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_archive_sha):
        raise ValueError("Missing valid pinned archive fingerprint")
    if sha256(archive) != expected_archive_sha:
        raise ValueError("Private Kaggle archive SHA-256 mismatch")
    with zipfile.ZipFile(archive) as z:
        records = z.infolist()
        if (len(records) != 3 or set(z.namelist()) !=
                {"neural_proof.mp4", "neural_proof.receipt.json", "technical_qc.json"}
                or any(i.is_dir() or i.file_size > 120_000_000 or
                       (i.external_attr >> 16) & 0o170000 == 0o120000
                       for i in records)):
            raise ValueError("Unexpected Kaggle video archive content")
        mp4 = z.read("neural_proof.mp4")
        proof = json.loads(z.read("neural_proof.receipt.json"))
        qc = json.loads(z.read("technical_qc.json"))
    if (not mp4.startswith(b"\x00\x00") or b"ftyp" not in mp4[:64]
            or hashlib.sha256(mp4).hexdigest() != receipt.get("clip_sha256")
            or proof.get("clip_sha256") != receipt["clip_sha256"]
            or qc.get("all_frames_decoded") is not True
            or qc.get("real_motion_detected") is not True):
        raise ValueError("Video invalid, stale or does not show independent motion")
    return {
        "status": "REAL_GPU_VIDEO_DECODED_REVIEW_REQUIRED", "model": "wan22-5b",
        "nonce": nonce, "video": receipt["video"], "sha256": receipt["clip_sha256"],
        "motion_delta": qc.get("first_last_mean_difference"),
        "human_approved": False, "published": False,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest="action", required=True)
    a = cmd.add_parser("prepare")
    a.add_argument("--username", required=True)
    a.add_argument("--nonce", required=True)
    a.add_argument("--destination", type=Path, required=True)
    b = cmd.add_parser("verify")
    b.add_argument("--path", type=Path, required=True)
    b.add_argument("--nonce", required=True)
    args = p.parse_args(argv)
    result = prepare(args.username, args.nonce, args.destination) if args.action == "prepare" else verify(args.path, args.nonce)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
