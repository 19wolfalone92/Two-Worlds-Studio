"""Single private Kaggle GPU proof: native ComfyUI Wan2.2 TI2V-5B, not a slideshow.

Generated from the audited Studio workflow. No private media or credentials in
this file; the only input picture is generated deterministically from Pillow
as a PUBLIC engineering fixture. No media is uploaded to public GitHub.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

CONFIG = "__TW_VIDEO_CONFIG__"
BUNDLE = "__TW_VIDEO_BUNDLE__"
WORK = Path("/kaggle/working")
STUDIO = Path("/kaggle/temp/two_worlds_wan_video")
COMFY = Path("/kaggle/temp/ComfyUI")
RECEIPT = WORK / "two_worlds_job_receipt.json"
ARCHIVE = WORK / "TWO_WORLDS_ARTIFACTS.zip"
SERVER_LOG = WORK / "wan_comfy_server.log"


def run(command, *, timeout=900, cwd=None, output=None):
    if output is None:
        subprocess.run(command, cwd=cwd, check=True, timeout=timeout)
    else:
        with output.open("wb") as stream:
            subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)


def source_import():
    STUDIO.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE)))
    for entry in archive.infolist():
        path = Path(entry.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in path.parts
                or entry.file_size > 150000):
            raise ValueError("Untrusted worker source path")
    archive.extractall(STUDIO)
    sys.path.insert(0, str(STUDIO))


def draw_fixture(path):
    from PIL import Image, ImageDraw, ImageFilter
    # A stylized synthetic figure; no personal or proprietary reference art.
    W, H = 576, 1024
    frame = Image.new("RGB", (W, H), (44, 52, 67))
    pic = ImageDraw.Draw(frame)
    for y in range(H):
        rgb = (43 + int(y / H * 12), 51 + int(y / H * 13), 65 + int(y / H * 8))
        pic.line([(0, y), (W, y)], fill=rgb)
    pic.rectangle([50, 105, 516, 960], outline=(105, 118, 123), width=5)
    pic.rectangle([340, 165, 490, 850], fill=(62, 72, 83))
    pic.ellipse([198, 186, 391, 390], fill=(70, 49, 42))
    pic.ellipse([225, 218, 364, 363], fill=(206, 151, 124))
    pic.polygon([(184, 366), (386, 364), (464, 920), (116, 920)], fill=(28, 34, 44))
    pic.ellipse([262, 283, 272, 291], fill=(44, 37, 36))
    pic.ellipse([316, 283, 326, 291], fill=(44, 37, 36))
    pic.arc([268, 310, 323, 340], 10, 160, fill=(99, 53, 51), width=3)
    pic.polygon([(190, 397), (116, 444), (85, 790), (150, 809), (233, 515)], fill=(31, 40, 52))
    pic.polygon([(380, 391), (462, 440), (480, 790), (419, 794), (349, 496)], fill=(32, 39, 50))
    pic.line([(100, 890), (475, 890)], fill=(120, 107, 90), width=6)
    frame.save(path)


def verify_motion(path, expected_frames):
    import subprocess
    from PIL import Image, ImageChops, ImageStat
    from studio.comfy_neural import _probe
    measured = _probe(path)
    if (measured["frames"] != expected_frames or measured["fps"] != "24"
            or measured["width"] != 448 or measured["height"] != 800):
        raise RuntimeError("Generated video does not have the required real frame clock")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror",
         "-i", str(path), "-f", "null", "-"], timeout=180)
    sample_dir = WORK / ".wan_frame_qc"
    sample_dir.mkdir(exist_ok=True)
    samples = [sample_dir / "start.png", sample_dir / "end.png"]
    for n, target in [(0, samples[0]), (expected_frames - 1, samples[1])]:
        run(["ffmpeg", "-v", "error", "-i", str(path),
             "-vf", "select=eq(n\\," + str(n) + ")", "-vsync", "vfr",
             "-frames:v", "1", str(target)], timeout=90)
        if not target.is_file() or target.stat().st_size < 500:
            raise RuntimeError("Missing fully decoded sample frame")
    with Image.open(samples[0]) as one, Image.open(samples[1]) as two:
        delta = float(ImageStat.Stat(ImageChops.difference(one.convert("RGB"), two.convert("RGB"))).mean[0])
    if delta < 0.75:
        raise RuntimeError("Neural output is effectively frozen (no independent motion)")
    return {
        "model": "wan22-5b", "video": measured, "all_frames_decoded": True,
        "first_last_mean_difference": round(delta, 4), "real_motion_detected": True,
        "human_review_required": True, "published": False
    }


def run_job(data):
    source_import()
    from PIL import Image
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle did not provide an NVIDIA CUDA GPU")
    device = {
        "name": torch.cuda.get_device_name(0),
        "memory_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    }
    if device["memory_total_gib"] < 14:
        raise RuntimeError("Need a real 16 GB-class NVIDIA T4 for this low VRAM proof")
    print("WAN_GPU_PREFLIGHT:", device, flush=True)

    run(["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git",
         str(COMFY)], timeout=420)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=COMFY, text=True).strip()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(COMFY / "requirements.txt"),
         "huggingface_hub>=0.29", "Pillow>=10,<12"], timeout=1000)

    from studio.model_install import install
    staged = install("wan22-5b", COMFY, accept_license=True, execute=True)
    if staged.get("status") != "WEIGHTS_STAGED_PRIVATELY_GPU_NOT_RUN":
        raise RuntimeError("Official model weights were not completely staged")
    reference = WORK / "wan_reference.png"
    draw_fixture(reference)
    server = None
    logfile = SERVER_LOG.open("wb")
    try:
        server = subprocess.Popen(
            [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188",
             "--disable-auto-launch", "--lowvram", "--cpu-vae"],
            cwd=COMFY, stdout=logfile, stderr=subprocess.STDOUT,
        )
        from studio.comfy_neural import ComfyLocal, generate
        client = ComfyLocal("http://127.0.0.1:8188", timeout=25)
        start = time.monotonic()
        while time.monotonic() - start < 240:
            if server.poll() is not None:
                raise RuntimeError("ComfyUI exited at startup; inspect private server log")
            try:
                client.json("/system_stats")
                break
            except Exception:
                time.sleep(3)
        else:
            raise TimeoutError("ComfyUI server did not become ready")
        output = WORK / "neural_proof.mp4"
        result = generate(
            "wan22-5b", reference=reference, output=output,
            prompt=data["prompt"], seed=data["seed"],
            host="http://127.0.0.1:8188", width=data["width"],
            height=data["height"], frames=data["frames"],
            steps=data["steps"], timeout=2700,
            accepted_license=True, execute=True, client=client,
        )
        if result.get("status") != "GENERATED_REVIEW_REQUIRED":
            raise RuntimeError("ComfyUI did not produce a real model MP4")
        qc = verify_motion(output, data["frames"])
        (WORK / "technical_qc.json").write_text(json.dumps(qc, indent=2) + "\n")
        output_receipt = output.with_suffix(".receipt.json")
        with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as zipped:
            zipped.write(output, arcname="neural_proof.mp4")
            zipped.write(output_receipt, arcname="neural_proof.receipt.json")
            zipped.write(WORK / "technical_qc.json", arcname="technical_qc.json")
        record = {
            "schema": 1, "nonce": data["nonce"], "success": True,
            "source_revision": data["source_revision"], "comfy_revision": revision,
            "model": "wan22-5b", "gpu": device,
            "weights_revision": staged["pinned_revisions"], "weights_staged": True,
            "inference_performed": True, "video": qc["video"],
            "clip_sha256": result["clip_sha256"],
            "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
            "human_approved": False, "published": False
        }
        RECEIPT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print("TW_REAL_WAN22_VIDEO_GENERATED_REVIEW_REQUIRED", flush=True)
        return record
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
        logfile.close()


def main():
    data = json.loads(base64.b64decode(CONFIG))
    try:
        if not (data.get("schema") == 1 and data.get("model") == "wan22-5b"
                and data.get("frames") == 17 and data.get("steps") == 20
                and [data.get("width"), data.get("height")] == [448, 800]):
            raise ValueError("Unreviewed GPU job parameters")
        run_job(data)
    except BaseException as exc:
        RECEIPT.write_text(json.dumps({
            "schema": 1, "nonce": data.get("nonce"), "success": False,
            "error_type": type(exc).__name__, "error_summary": str(exc)[:240],
            "inference_performed": False, "published": False
        }, indent=2) + "\n")
        print("TW_REAL_WAN22_VIDEO_FAILED:", type(exc).__name__, str(exc)[:180], flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
