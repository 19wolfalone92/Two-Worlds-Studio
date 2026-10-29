"""Kaggle-side single-run worker, bundled through reviewed GitHub Actions.

This file is a TEMPLATE, never executable until the official build helper
injects source and job metadata. No API credentials are sent to Kaggle.
No screenplay, private reference photo, or GitHub token is bundled.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

CONFIG_BASE64 = "__TW_CONFIG_B64__"
SOURCE_BASE64 = "__TW_BUNDLE_B64__"
WORKING = Path("/kaggle/working")
TEMP = Path("/kaggle/temp/two_worlds_modules")
OUTPUT = WORKING / "two_worlds_outputs"
RECEIPT = WORKING / "two_worlds_job_receipt.json"
ARCHIVE = WORKING / "TWO_WORLDS_ARTIFACTS.zip"

def safe_unpack(data: bytes, target: Path):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for item in archive.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("Invalid uploaded source path")
        archive.extractall(target)

def install(*packages):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages],
                   check=True, timeout=1800)

def run_job(config):
    safe_unpack(base64.b64decode(SOURCE_BASE64), TEMP)
    sys.path.insert(0, str(TEMP))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mode = config["mode"]
    result = {}
    if mode=="design":
        install("git+https://github.com/huggingface/diffusers.git",
                "transformers", "accelerate", "safetensors", "sentencepiece", "Pillow", "numpy")
        from studio.character_factory import create_design,run_batch,FluxBackend
        spec_file=TEMP/"new_character_design.json"
        spec_file.write_text(json.dumps(config["design_spec"],ensure_ascii=False))
        project=OUTPUT/"character_factory"
        create_design(spec_file,project)
        result["design"]=run_batch(project,FluxBackend(project,width=768,height=1152),
                    max_images=config["max_images"],views=("front","three_quarter","profile"))
    if mode in ("portrait", "both"):
        install("git+https://github.com/huggingface/diffusers.git",
                "transformers", "accelerate", "safetensors", "sentencepiece")
        from studio.kaggle_flux import character_job, run_gpu
        from studio.art_director import load_style
        job = character_job(load_style(), config["body"], config["view"])
        filename = f'{config["body"]}_{config["view"]}.png'
        result["portrait"] = run_gpu(job, OUTPUT / filename,
                                      width=576, height=1024, steps=4)
    if mode in ("voice", "both"):
        install("kokoro>=0.9.2", "soundfile")
        if subprocess.run(["bash", "-lc", "command -v espeak-ng"],
                          stdout=subprocess.DEVNULL).returncode:
            subprocess.run(["bash", "-lc",
               "apt-get update -qq && apt-get install -y -qq espeak-ng"],
               check=True, timeout=900)
        from studio.kokoro_tts import synthesize, load_cast
        cast = load_cast()
        preset = cast["speakers"]["Mira"]
        voice = synthesize(
            "This is a technical voice sample for Two Worlds Studio.",
            preset["voice"], preset["language"], OUTPUT / "Mira_voice_sample.wav")
        result["voice"] = voice
    return result

def archive_output(config, result, *, success=True):
    manifest = {
        "schema": 1, "run_nonce": config["run_nonce"],
        "source_sha": config["source_sha"], "mode": config["mode"],
        "success":success,
        "models_reviewed": False, "art_approved": False,
        "voices_approved": False, "result": result,
    }
    (OUTPUT / "output_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
        for file in sorted(OUTPUT.rglob("*")):
            if not file.is_file() or file.suffix.lower() not in (".png", ".wav", ".json"):
                continue
            z.write(file, arcname=file.relative_to(OUTPUT).as_posix())
    return {
        "run_nonce": config["run_nonce"], "success": success,
        "source_sha": config["source_sha"], "mode": config["mode"],
        "result": result,
        "archive": ARCHIVE.name,
        "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
    }

def main():
    config = json.loads(base64.b64decode(CONFIG_BASE64).decode("utf-8"))
    try:
        result = run_job(config)
        receipt = archive_output(config, result)
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print("TWO WORLDS Kaggle generation succeeded. Outputs are UNREVIEWED.")
    except BaseException as exc:
        receipt = {"run_nonce": config.get("run_nonce"), "success": False,
                   "source_sha":config.get("source_sha"),"mode":config.get("mode"),
                   "error_type": type(exc).__name__,
                   "message": str(exc)[:600]}
        # Preserve completed drawings and their ledger on a later GPU failure.
        # They remain explicit partial results, never a successful batch.
        if OUTPUT.exists():
            try:
                ledger=OUTPUT/"character_factory/ledger.json"
                count=len(json.loads(ledger.read_text()).get("results",[])) if ledger.is_file() else 0
                receipt.update(archive_output(config,{"partial":True,"completed_drawings":count},success=False))
                receipt.update(error_type=type(exc).__name__,message=str(exc)[:600])
            except Exception:
                pass
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print("TWO WORLDS Kaggle generation failed (see actual Kaggle logs).")
        raise

if __name__ == "__main__":
    main()
