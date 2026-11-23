"""PRIVATE Kaggle GPU worker template for the isolated Freevideo hybrid.

Markers are filled ONLY by kaggle_private_job.prepare, which embeds the photo
into a locally generated private Kaggle kernel script, NEVER the public repo.
This worker downloads only open-source ComfyUI and official licensed Wan
checkpoint components, then executes local inference via Freevideo + Studio.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

CONFIG_B64 = "__FREEVIDEO_JOB_CONFIG_B64__"
IMAGE_B64 = "__FREEVIDEO_PRIVATE_IMAGE_B64__"
WORK = Path("/kaggle/working")
ROOT = Path("/kaggle/temp/freevideo_public_source")
COMFY = Path("/kaggle/temp/freevideo_comfy")
OUT = WORK / "freevideo_result.json"
ARCHIVE = WORK / "FREEVIDEO_PRIVATE_PREVIEW.zip"


def run(command: list[str], *, timeout: int = 900, cwd: Path | None = None):
    """No shell, no ambient publishing, finite runtime."""
    return subprocess.run(command, check=True, timeout=timeout, cwd=cwd)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_private_source(config: dict) -> Path:
    fmt = config.get("reference_format")
    if fmt not in ("png", "jpeg"):
        raise ValueError("Expected inspected PNG or JPEG format in private config")
    suffix = ".png" if fmt == "png" else ".jpg"
    work = WORK / ("freevideo_source_input" + suffix)
    if work.exists():
        raise ValueError("Output collision; do not overwrite a private reference")
    data = base64.b64decode(IMAGE_B64, validate=True)
    if not 1024 <= len(data) <= 32 * 1024 * 1024:
        raise ValueError("Private reference size outside permitted bounds")
    if (fmt == "png" and data[:8] != bytes.fromhex("89504e470d0a1a0a")) or (
            fmt == "jpeg" and data[:3] != bytes.fromhex("ffd8ff")):
        raise ValueError("Embedded photo bytes mismatch inspected file format")
    if hashlib.sha256(data).hexdigest() != config["reference_sha256"]:
        raise ValueError("Private reference SHA mismatch")
    with work.open("xb") as f:
        f.write(data)
    return work


def _checkout_source(config: dict) -> None:
    if ROOT.exists():
        raise ValueError("Unexpected public source checkout preexists")
    branch = "research/freevideo-t4-adaptive-20261123"
    run(["git", "clone", "--depth", "1", "--single-branch", "--branch", branch,
         "https://github.com/19wolfalone92/Two-Worlds-Studio.git", str(ROOT)], timeout=360)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True, cwd=ROOT, timeout=20).strip()
    if revision != config["source_revision"]:
        raise ValueError("Experimental Git SHA drift; rebuild PRIVATE job from reviewed branch")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "experiments/freevideo_t4"))


def _install_backend(config: dict) -> dict:
    if COMFY.exists():
        raise ValueError("Unexpected existing ComfyUI; cannot validate provenance")
    run(["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git",
         str(COMFY)], timeout=420)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True, cwd=COMFY, timeout=20).strip()
    run([sys.executable, "-m", "pip", "install", "-q", "-r",
         str(COMFY / "requirements.txt"), "huggingface_hub>=0.29", "Pillow>=10,<12"],
        timeout=1200)
    from studio.model_install import install
    details = install("wan22-5b", COMFY, accept_license=config["license_reviewed"],
                      execute=True)
    if details.get("status") != "WEIGHTS_STAGED_PRIVATELY_GPU_NOT_RUN":
        raise RuntimeError("Official checkpoint staging did not complete")
    return {"comfy_sha": revision, "weights_revision": details["pinned_revisions"]}


def _weight_manifest() -> Path:
    from studio.model_install import inventory
    inventory_result = inventory("wan22-5b", COMFY)
    components = {}
    for f in inventory_result["files"]:
        path = Path(f["target"])
        if not path.is_file() or path.is_symlink():
            raise ValueError("Required official weight missing or unsafe")
        components[path.name] = {"path": str(path), "sha256": sha256(path)}
    manifest = {
        "schema": 1, "model": "wan22-5b", "license_reviewed": True,
        "source_repository": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "files": components}
    location = WORK / "private_model_manifest.json"
    with location.open("x", encoding="utf-8") as f:
        json.dump(manifest, f)
    return location


def _start_comfy() -> tuple[subprocess.Popen, object]:
    log = (WORK / "freevideo_private_comfy.log").open("xb")
    process = subprocess.Popen([
        sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188",
        "--disable-auto-launch", "--lowvram", "--cpu-vae"
    ], cwd=COMFY, stdout=log, stderr=subprocess.STDOUT)
    try:
        from studio.comfy_neural import ComfyLocal
        client = ComfyLocal("http://127.0.0.1:8188", timeout=15)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("ComfyUI process exited; inspect PRIVATE server log")
            try:
                client.json("/system_stats")
                return process, log
            except Exception:
                time.sleep(3)
        raise TimeoutError("Local ComfyUI startup timeout")
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
        raise


def _gpu() -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_NOT_TESTED: Kaggle runtime has no CUDA GPU")
    details = torch.cuda.get_device_properties(0)
    if "T4" not in details.name or details.major != 7 or details.minor != 5:
        raise RuntimeError("This experimental worker permits only NVIDIA T4 compute 7.5")
    with torch.cuda.device(0):
        free, total = torch.cuda.mem_get_info()
    return {"name": details.name, "capability": "7.5",
            "free_gib": round(free / 2**30, 3), "total_gib": round(total / 2**30, 3)}


def run_job(data: dict) -> dict:
    if (data.get("schema") != 1 or data.get("model") != "wan22-5b"
            or data.get("license_reviewed") is not True or
            not isinstance(data.get("source_revision"), str) or
            not isinstance(data.get("reference_sha256"), str) or
            not isinstance(data.get("nonce"), str)):
        raise ValueError("Private Freevideo configuration missing licensed model fields")
    gpu = _gpu()
    _checkout_source(data)
    original = _load_private_source(data)
    from experiments.freevideo_t4.hybrid_engine import prepare, run_stage
    private_work = WORK / ("freevideo_job_" + data["nonce"])
    setup = prepare(original, private_work, data["nonce"], data["prompt"],
                    seed=data["seed"])
    backend = _install_backend(data)
    weights = _weight_manifest()
    process = log = None
    try:
        process, log = _start_comfy()
        result = run_stage(setup["job_file"], "preview",
                           execute=True, accept_model_license=True,
                           weights_manifest=str(weights), device=0,
                           host="http://127.0.0.1:8188", timeout=2700)
        if result.get("status") != "GENERATED_REVIEW_REQUIRED":
            raise RuntimeError("Native GPU I2V did not generate a validated preview")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        if log is not None:
            log.close()

    clip = private_work / "preview.mp4"
    files = [
        "preview.mp4", "preview.receipt.json", "preview.freevideo.json",
        "preview.hybrid.json", "job.json", "timings.jsonl"
    ]
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3) as package:
        for filename in files:
            p = private_work / filename
            if p.is_file():
                package.write(p, arcname=filename)
    record = {
        "schema": 1, "nonce": data["nonce"], "source_revision": data["source_revision"],
        "source_sha256": data["reference_sha256"],
        "archive_sha256": sha256(ARCHIVE), "clip_sha256": sha256(clip),
        "video_stage": "preview", "gpu": gpu, "backend_revisions": backend,
        "real_neural_backend": "studio.comfy_neural/wan22-5b",
        "status": "REAL_GPU_PREVIEW_REVIEW_REQUIRED",
        "human_approved": False, "published": False,
        "visual_quality": "NOT_VERIFIED",
        "note": "Original private photo not included in output archive."
    }
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


def main():
    data = json.loads(base64.b64decode(CONFIG_B64, validate=True))
    try:
        result = run_job(data)
        print(json.dumps({"status": result["status"],
                          "nonce": result["nonce"], "published": False}), flush=True)
    except BaseException as exc:
        record = {"schema": 1, "nonce": data.get("nonce"),
                  "status": "BLOCKED_GPU_NOT_VERIFIED", "error_type": type(exc).__name__,
                  "error_summary": str(exc)[:220], "published": False}
        with OUT.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
