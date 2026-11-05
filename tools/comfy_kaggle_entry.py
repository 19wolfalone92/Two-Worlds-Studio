"""Private Kaggle worker: actual ComfyUI server + SDXL + studio.art_director.

This file is a template; the packaging step replaces the two marker strings.
Only a single historical PUBLIC ENGINEERING FIXTURE scene is rendered. Neither
Kaggle credentials nor unpublished Work screenplay/character refs are sent.
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
import urllib.request
import zipfile
from pathlib import Path

CONFIG_B64 = "__TW_COMFY_CONFIG__"
SOURCE_B64 = "__TW_COMFY_SOURCE__"
WORK = Path("/kaggle/working")
STUDIO = Path("/kaggle/temp/two_worlds_comfy_source")
COMFY = Path("/kaggle/temp/ComfyUI")
OUTPUT = WORK / "two_worlds_comfy_outputs"
ARCHIVE = WORK / "TWO_WORLDS_ARTIFACTS.zip"
RECEIPT = WORK / "two_worlds_job_receipt.json"
SERVER_LOG = WORK / "comfy_server.log"


def install(*packages):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        check=True, timeout=1700,
    )


def unpack_source():
    STUDIO.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as z:
        for entry in z.infolist():
            path = Path(entry.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid relative module path in source bundle")
        z.extractall(STUDIO)
    sys.path.insert(0, str(STUDIO))


def gpu_preflight():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle did not allocate a CUDA GPU")
    details = {
        "gpu": torch.cuda.get_device_name(0),
        "cuda_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
    }
    print("GPU allocated:", details, flush=True)
    return details


def run_job(config):
    unpack_source()
    hardware = gpu_preflight()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # ComfyUI is an ON-DEMAND service in this one notebook session, not a
    # publicly exposed daemon. No proxy/tunnel, webhook or arbitrary URL.
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git", str(COMFY)],
        check=True, timeout=420,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=COMFY, text=True
    ).strip()
    install("-r", str(COMFY / "requirements.txt"))
    install("huggingface_hub", "Pillow>=10,<12")

    from huggingface_hub import hf_hub_download
    checkpoint_path = hf_hub_download(
        repo_id=config["checkpoint_repo"],
        filename=config["checkpoint_file"],
        local_dir=str(COMFY / "models" / "checkpoints"),
    )
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000_000:
        raise RuntimeError("Checkpoint download missing/truncated")
    print("SDXL checkpoint available; starting local ComfyUI.", flush=True)

    server = None
    logfile = SERVER_LOG.open("w", encoding="utf-8")
    try:
        server = subprocess.Popen(
            [
                sys.executable, "main.py", "--listen", "127.0.0.1",
                "--port", "8188", "--disable-auto-launch", "--lowvram",
            ],
            cwd=COMFY, stdout=logfile, stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 240
        alive = False
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("ComfyUI exited during startup; see comfy_server.log")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=4) as resp:
                    if resp.status == 200:
                        alive = True
                        break
            except Exception:
                time.sleep(3)
        if not alive:
            raise TimeoutError("ComfyUI did not initialize within startup deadline")

        from studio import art_director
        fixture = json.loads((STUDIO / config["episode_fixture"]).read_text(encoding="utf-8"))
        scene = next((x for x in fixture["scenes"] if x["id"] == config["scene"]), None)
        if scene is None:
            raise ValueError("Missing public test scene")
        shot = art_director.panel_prompt(scene, art_director.load_style(), fixture["id"])
        workflow_file = STUDIO / "templates/comfyui_sdxl_api.json"
        workflow = art_director.comfy_workflow(
            shot, json.loads(workflow_file.read_text(encoding="utf-8")),
            config["checkpoint_file"], width=config["width"], height=config["height"],
        )
        rendered = art_director.comfy_render(
            workflow, "http://127.0.0.1:8188", OUTPUT / "render.png", timeout=1100
        )
        png_path = OUTPUT / "render.png"
        result = {
            "schema": 1, "source_sha": config["source_sha"],
            "comfy_revision": revision,
            "comfy_prompt_id": rendered["comfy_prompt_id"],
            "checkpoint_repo": config["checkpoint_repo"],
            "checkpoint_file": config["checkpoint_file"],
            "width": config["width"], "height": config["height"],
            "scene": config["scene"], "seed": shot["seed"],
            "gpu": hardware, "png_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
            "review_status": "unreviewed", "identity_verified": False,
            "no_publication": True, "no_reference_conditioning": True,
        }
        (OUTPUT / "render_manifest.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return result
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
        logfile.close()


def main():
    config = json.loads(base64.b64decode(CONFIG_B64))
    try:
        output = run_job(config)
        with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(OUTPUT / "render.png", arcname="render.png")
            archive.write(OUTPUT / "render_manifest.json", arcname="render_manifest.json")
        receipt = {
            "nonce": config["nonce"], "success": True,
            "source_sha": config["source_sha"],
            "comfy_prompt_id": output["comfy_prompt_id"],
            "archive": ARCHIVE.name, "render_sha256": output["png_sha256"],
            "review_status": "unreviewed",
        }
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print("COMFYUI_REAL_IMAGE_GENERATED_UNREVIEWED", flush=True)
    except BaseException as exc:
        RECEIPT.write_text(json.dumps({
            "nonce": config.get("nonce"), "success": False,
            "error_type": type(exc).__name__, "error": str(exc)[:500],
        }, indent=2) + "\n", encoding="utf-8")
        print("COMFYUI_RUN_FAILED:", type(exc).__name__, str(exc)[:300], flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
