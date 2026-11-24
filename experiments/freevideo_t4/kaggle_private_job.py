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
import tempfile
import zipfile

from PIL import Image

try:
    from .hybrid_engine import ROOT, private_path, sha256, HybridBlocked
except ImportError:
    from hybrid_engine import ROOT, private_path, sha256, HybridBlocked

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
    if image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise KaggleBlocked("Use an existing private PNG/JPEG photo as Kaggle reference")
    if image.stat().st_size > 8 * 1024 * 1024:
        raise KaggleBlocked("For the private kernel embed, source PNG must be <=8 MiB")
    try:
        with Image.open(image) as pic:
            reference_format = pic.format
            pic.verify()
    except OSError as exc:
        raise KaggleBlocked("Private photo cannot be decoded") from exc
    if reference_format not in ("PNG", "JPEG"):
        raise KaggleBlocked("Only genuinely decoded PNG/JPEG bytes are supported")
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
        "reference_format": reference_format.lower(),
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


def verify_private_result(folder: str | Path, nonce: str, stage: str = "preview") -> dict:
    """Validate returned Kaggle archive without trusting declared success."""
    location = Path(folder).expanduser().absolute()
    if not NONCE.fullmatch(nonce):
        raise KaggleBlocked("Invalid nonce")
    if stage not in ("preview", "candidate"):
        raise KaggleBlocked("Unsupported private stage")
    manifest = private_path(location / ("freevideo_result.json" if stage == "preview" else "freevideo_candidate_result.json"))
    archive = private_path(location / ("FREEVIDEO_PRIVATE_" + stage.upper() + ".zip"))
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("nonce") != nonce
            or record.get("status") != "REAL_GPU_" + stage.upper() + "_REVIEW_REQUIRED"
            or record.get("human_approved") is not False
            or record.get("published") is not False
            or record.get("visual_quality") != "NOT_VERIFIED"):
        raise KaggleBlocked("No matching successful GPU receipt")
    if not 8000 <= archive.stat().st_size <= 300 * 1024 * 1024:
        raise KaggleBlocked("Private preview archive size invalid")
    if sha256(archive) != record.get("archive_sha256"):
        raise KaggleBlocked("Kaggle ZIP SHA mismatch")
    wanted = {stage + suffix for suffix in (".mp4", ".receipt.json", ".freevideo.json", ".hybrid.json")}
    wanted.add("job.json")
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        names = [m.filename for m in members]
        if (len(names) != len(set(names)) or not wanted.issubset(names)
                or set(names) - wanted - {"timings.jsonl"}
                or any(m.is_dir() or m.file_size > 300 * 1024 * 1024
                       or (m.external_attr >> 16) & 0o170000 == 0o120000
                       for m in members)):
            raise KaggleBlocked("Unsafe/missing private preview ZIP members")

        raw = z.read(stage + ".mp4")
        studio_bytes = z.read(stage + ".receipt.json")
        freevideo_bytes = z.read(stage + ".freevideo.json")
        studio = json.loads(studio_bytes)
        hybrid = json.loads(z.read(stage + ".hybrid.json"))
        freevideo = json.loads(freevideo_bytes)
        job = json.loads(z.read("job.json"))
        digest = hashlib.sha256(raw).hexdigest()
        if (digest != record.get("clip_sha256")
                or digest != studio.get("clip_sha256")
                or digest != hybrid.get("clip_sha256")
                or digest != freevideo.get("clip_sha256")
                or hybrid.get("stage") != stage
                or studio.get("schema") != 1 or studio.get("engine") != "neural_comfy"
                or studio.get("model") != "wan22-5b"
                or studio.get("inference_performed") is not True
                or studio.get("human_approved") is not False
                or studio.get("published") is not False
                or freevideo.get("schema") != 2
                or freevideo.get("stage") != stage
                or freevideo.get("bridge_receipt_sha256")
                   != hashlib.sha256(studio_bytes).hexdigest()
                or hybrid.get("studio_receipt_sha256")
                   != hashlib.sha256(studio_bytes).hexdigest()
                or hybrid.get("freevideo_receipt_sha256")
                   != hashlib.sha256(freevideo_bytes).hexdigest()
                or record.get("source_sha256") != job.get("source_sha256")
                or hybrid.get("source_sha256") != job.get("source_sha256")
                or hybrid.get("reference_sha256") != job.get("reference_sha256")):
            raise KaggleBlocked("Native MP4/Studio/Freevideo provenance mismatch")
        measured_receipt = studio.get("video", {})
        expected_frames = 25 if stage == "preview" else 121
        allowed = (((432, 576), (336, 448)) if stage == "preview" else ((528, 704), (384, 512))) if job.get("framing") == "native_3x4" else ((432, 768), (288, 512))
        if (not isinstance(measured_receipt, dict)
                or measured_receipt.get("frames") != expected_frames
                or (measured_receipt.get("width"), measured_receipt.get("height")) not in allowed
                or measured_receipt.get("fps") != "24"):
            raise KaggleBlocked("Unexpected GPU preview frame-clock receipt")
    # Hashes and JSON are not video decoding. Actually decode every MP4 frame.
    try:
        from quality_metrics import audit_video
    except ImportError:
        from .quality_metrics import audit_video
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "kaggle_preview.mp4"
        with p.open("xb") as stream:
            stream.write(raw)
        try:
            qc = audit_video(p)
        except (OSError, ValueError) as exc:
            raise KaggleBlocked("Returned Kaggle clip does not fully decode") from exc
    if (qc["video_sha256"] != digest or qc["frames"] != expected_frames
            or qc["width"] != measured_receipt["width"] or qc["height"] != measured_receipt["height"] or qc["fps"] != "24"):
        raise KaggleBlocked("Returned MP4 decoded frame-clock differs from Studio")
    return {"status": "REMOTE_GPU_RECEIPT_MP4_DECODED_REVIEW_REQUIRED",
            "clip_sha256": record["clip_sha256"], "nonce": nonce, "stage": stage,
            "actual_decoding": "CPU_VERIFIED",
            "neural_provenance": "REMOTE_KAGGLE_STUDIO_RECEIPT",
            "visual_quality": "NOT_VERIFIED", "temporal_flags": qc["flags"],
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
    b.add_argument("--stage", choices=("preview", "candidate"), default="preview")
    v = p.parse_args(argv)
    try:
        if v.cmd == "prepare":
            result = prepare_private_kernel(v.username, v.nonce, v.source,
                                            v.prompt, v.destination,
                                            revision=v.revision, seed=v.seed,
                                            accept_model_license=v.accept_model_license)
        else:
            result = verify_private_result(v.path, v.nonce, v.stage)
        code = 0
    except (KaggleBlocked, OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        result, code = {"status": "BLOCKED", "reason": str(exc),
                        "gpu_success": False, "published": False}, 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
