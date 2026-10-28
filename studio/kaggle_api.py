"""Prepare and orchestrate a private Kaggle kernel from reviewed GitHub Actions.

Uses official Kaggle CLI kernels push/status/output. Never stores API tokens.
GPU runs only after manually dispatched workflow and secured GitHub secret.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLUG = "two-worlds-visual-gpu"
USERNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,29}$")
BODY = {"earth_mira", "noctaris_lyra", "earth_eli", "noctaris_ilan"}
VIEWS = {"front", "three_quarter", "profile", "full_body", "expression"}
MODES = {"portrait", "voice", "both"}
STATUS = re.compile(r"KERNELWORKERSTATUS\.([A-Z_]+)")
RECEIPT = "two_worlds_job_receipt.json"
PACKAGE = "TWO_WORLDS_ARTIFACTS.zip"
FILES = (
    "studio/__init__.py", "studio/kaggle_flux.py",
    "studio/kokoro_tts.py", "studio/art_director.py",
    "studio/identity_qc.py", "studio/render.py",
    "studio/continuity.py", "studio/premium_assets.py",
    "studio/comic_art.py", "studio/cinematic.py",
    "studio/actor_motion.py", "studio/audio_master.py",
    "studio/lipsync.py", "story/visual_bible.json",
    "story/identity_registry.json", "audio/kokoro_cast.json",
    "audio/voice_registry.json",
)

def package_source(root: Path = ROOT) -> bytes:
    """Embed only explicitly audited modules, not secrets or notebooks."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in FILES:
            file = (root / relative).resolve()
            if not file.is_file() or root.resolve() not in file.parents:
                raise FileNotFoundError(f"Missing required module {relative}")
            if file.stat().st_size > 300_000:
                raise ValueError(f"Unusually large module {relative}")
            archive.write(file, arcname=relative)
    if buf.tell() > 1_500_000:
        raise ValueError("Reviewed source bundle exceeded 1.5 MB")
    return buf.getvalue()

def build_job(username: str, nonce: str, mode: str = "portrait",
              body: str = "earth_mira", view: str = "front",
              *, source_root: Path = ROOT, output_dir: Path,
              git_sha: str | None = None) -> dict:
    if not USERNAME.fullmatch(username):
        raise ValueError("Kaggle username must be 2–30 safe characters")
    if not isinstance(nonce, str) or not re.fullmatch(r"[A-Za-z0-9_-]{3,90}", nonce):
        raise ValueError("Invalid run nonce")
    if mode not in MODES or body not in BODY or view not in VIEWS:
        raise ValueError("Unknown generation job parameters")
    if git_sha is None:
        git_sha = os.environ.get("GITHUB_SHA", "local-dry-run")
    if not isinstance(git_sha, str) or len(git_sha) > 80:
        raise ValueError("Invalid source revision")
    payload = {"schema": 1, "run_nonce": nonce, "mode": mode, "body": body,
               "view": view, "source_sha": git_sha}
    bundle = package_source(source_root)
    template = (source_root / "tools/kaggle_api_kernel_entry.py").read_text(encoding="utf-8")
    if template.count("__TW_CONFIG_B64__") != 1 or template.count("__TW_BUNDLE_B64__") != 1:
        raise RuntimeError("Incorrect Kaggle entrypoint template placeholders")
    config = base64.b64encode(json.dumps(payload, sort_keys=True).encode()).decode("ascii")
    data = base64.b64encode(bundle).decode("ascii")
    rendered = template.replace("__TW_CONFIG_B64__", config).replace("__TW_BUNDLE_B64__", data)
    kernel = f"{username}/{SLUG}"
    meta = {
        "id": kernel, "title": "Two Worlds Visual GPU",
        "code_file": "kernel_main.py", "language": "python",
        "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true" if mode != "voice" else "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4" if mode != "voice" else "",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "kernel_main.py").write_text(rendered, encoding="utf-8")
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"kernel": kernel, "run_nonce": nonce, "mode": mode,
            "metadata": meta, "source_bundle_bytes": len(bundle),
            "source_sha256": hashlib.sha256(bundle).hexdigest(),
            "credential_included": False}

def parse_status(stdout: str) -> str:
    match = STATUS.search(stdout.upper())
    if match is None:
        raise ValueError("Unknown Kaggle status response; cannot infer success")
    return match.group(1)

def poll_kernel(kernel: str, *, timeout_seconds: int = 3300,
                interval_seconds: int = 45, runner=subprocess.run) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[a-z0-9-]+", kernel):
        raise ValueError("Invalid Kaggle kernel handle")
    if not 15 <= interval_seconds <= 300 or not 60 <= timeout_seconds <= 5400:
        raise ValueError("Polling limits out of bounds")
    deadline = time.monotonic() + timeout_seconds
    last = ""
    while time.monotonic() < deadline:
        proc = runner(["kaggle", "kernels", "status", kernel], check=True,
                      capture_output=True, text=True, timeout=60)
        status = parse_status(proc.stdout)
        if status != last:
            print(f"Kaggle status: {status}", flush=True)
            last = status
        if status == "COMPLETE":
            return status
        if status in {"ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"}:
            raise RuntimeError(f"Kaggle kernel failed: {status}")
        if status not in {"QUEUED", "RUNNING", "NEW_SCRIPT"}:
            raise RuntimeError(f"Unknown Kaggle kernel lifecycle: {status}")
        time.sleep(interval_seconds)
    raise TimeoutError("Kaggle did not finish within approved workflow deadline")

def verify_outputs(path: Path, nonce: str) -> dict:
    receipts = list(path.rglob("two_worlds_job_receipt.json"))
    archives = list(path.rglob("TWO_WORLDS_ARTIFACTS.zip"))
    if len(receipts) != 1 or len(archives) != 1:
        raise FileNotFoundError("Kaggle output archive or receipt is missing")
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    if receipt.get("run_nonce") != nonce:
        raise ValueError("Result is from another run; refuse to mix outputs")
    if receipt.get("success") is not True:
        raise RuntimeError("Remote Kaggle job did not succeed")
    with zipfile.ZipFile(archives[0]) as z:
        entries = z.namelist()
        if not entries or any(Path(n).is_absolute() or ".." in Path(n).parts for n in entries):
            raise ValueError("Unsafe or empty output archive")
        if not any(n.lower().endswith((".png", ".wav")) for n in entries):
            raise ValueError("Kaggle did not generate an image or voice clip")
    return {"run_nonce": nonce, "success": True, "files": len(entries),
            "archive_bytes": archives[0].stat().st_size,
            "sha256": hashlib.sha256(archives[0].read_bytes()).hexdigest()}

def main():
    p = argparse.ArgumentParser(description="Manual private Kaggle API runner")
    cmd = p.add_subparsers(dest="action", required=True)
    create = cmd.add_parser("prepare")
    create.add_argument("--username", required=True)
    create.add_argument("--run-nonce", required=True)
    create.add_argument("--mode", choices=sorted(MODES), default="portrait")
    create.add_argument("--body", choices=sorted(BODY), default="earth_mira")
    create.add_argument("--view", choices=sorted(VIEWS), default="front")
    create.add_argument("--output-dir", type=Path, required=True)
    poll = cmd.add_parser("poll")
    poll.add_argument("--kernel", required=True)
    poll.add_argument("--timeout", type=int, default=3300)
    poll.add_argument("--interval", type=int, default=45)
    verify = cmd.add_parser("verify")
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument("--run-nonce", required=True)
    a = p.parse_args()
    if a.action == "prepare":
        result = build_job(a.username, a.run_nonce, a.mode, a.body, a.view,
                           output_dir=a.output_dir)
    elif a.action == "poll":
        result = {"status": poll_kernel(a.kernel, timeout_seconds=a.timeout,
                                        interval_seconds=a.interval)}
    else:
        result = verify_outputs(a.output_dir, a.run_nonce)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
