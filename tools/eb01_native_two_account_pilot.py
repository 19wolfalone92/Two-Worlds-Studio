"""One bounded private native pilot per independently authorized Kaggle account.

No production dispatcher. Reruns collect the same immutable notebook; never
resubmit an uncertain push. Raw output is returned only through RSA/AES-GCM.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

from studio.eb01_native_t4_profile import PROFILE, BRANCH
from studio.sealed_artifacts import seal
from tools.eb01_kaggle2_auth_probe import query, SOURCE_DATASET
from tools.eb01_kaggle_token_env import token_environment
from tools.eb01_private_two_account_dataset_sync import classify_error

REPO = "https://github.com/19wolfalone92/Two-Worlds-Studio.git"
STATE = re.compile(r"KernelWorkerStatus\.(COMPLETE|QUEUED|RUNNING|ERROR|CANCELLED|CANCEL_ACKNOWLEDGED)")
SHOTS = {"owner": "S02", "collaborator": "S01"}


class PilotBlocked(RuntimeError):
    pass


def call(token: str, args: list[str], timeout=120) -> str:
    with token_environment(token) as env:
        try:
            p = subprocess.run(args, env=env, capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            raise PilotBlocked("CLI_UNAVAILABLE_OR_TIMEOUT") from None
    if p.returncode:
        raise PilotBlocked(classify_error((p.stdout or "") + "\n" + (p.stderr or "")))
    return p.stdout or ""


def prepare(root: Path, alias: str, handle: str, run_id: str, revision: str) -> tuple[Path, str]:
    if alias not in SHOTS or not re.fullmatch(r"[1-9][0-9]{5,20}", run_id):
        raise PilotBlocked("INVALID_ACCOUNT_OR_RUN")
    if not re.fullmatch(r"[a-f0-9]{40}", revision) or not re.fullmatch(r"[a-zA-Z0-9_-]{2,30}", handle):
        raise PilotBlocked("INVALID_PINNED_IDENTITY")
    nonce = hashlib.sha256(("eb01-native-v1-"+run_id+"-"+alias).encode()).hexdigest()[:12]
    slug = "eb01-native-pilot-" + nonce
    cfg = {"schema": 7, "shot": SHOTS[alias], "private": True,
           "no_paid_services": True, "new_gpu_jobs": 1, "allow_social_publish": False,
           "mode": "pilot", "pilot_frames": 33, "pilot_profile": PROFILE,
           "source_dataset": SOURCE_DATASET, "branch": BRANCH,
           "revision": revision, "nonce": nonce}
    source = Path("tools/kaggle_eb01_ltx2b_s02_fallback.py").read_text()
    tag = "__EB01_LTX2B_CONFIG_B64__"
    if source.count(tag) != 1:
        raise PilotBlocked("AMBIGUOUS_KERNEL_CONFIG")
    source = source.replace(tag, base64.b64encode(json.dumps(cfg, sort_keys=True).encode()).decode())
    dest = root / "kernel"
    dest.mkdir(mode=0o700)
    (dest / "kernel_main.py").write_text(source)
    kernel = handle + "/" + slug
    metadata = {"id": kernel, "title": slug, "code_file": "kernel_main.py",
                "language": "python", "kernel_type": "script", "is_private": "true",
                "enable_gpu": "true", "enable_internet": "true",
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [SOURCE_DATASET],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (dest / "kernel-metadata.json").write_text(json.dumps(metadata))
    return dest, kernel


def verify_output(output: Path, alias: str, revision: str) -> dict:
    """Independently decode actual bytes; a kernel receipt alone cannot pass."""
    from studio.comfy_neural import _probe
    from studio.eb01_motion_artifact_gate import inspect_clip
    from tools.eb01_true_gpu_clip_numbers import gray_motion
    video = output / "S02_PILOT33.mp4"
    result = {"actual_mp4_created": video.is_file(), "technical_pass": False,
              "actor_motion_approved": False, "audio_conditioned_lipsync_verified": False}
    receipt_path = output / "EB01_S02_LTX_PILOT33_JOB.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    if not video.is_file():
        result["error_type"] = receipt.get("error_type", "NO_NATIVE_MP4")
        log = output / "EB01_LTX_PRIVATE_GPU_LOG.txt"
        raw = log.read_text(errors="replace") if log.is_file() else ""
        # Only predefined classes leave the private directory.
        result["failure_category"] = (
            "CUDA_OUT_OF_MEMORY" if "out of memory" in raw.lower() else
            "DECODER_TIMESTEP_OR_TILE_ERROR" if "timestep" in raw.lower() and "error" in raw.lower() else
            "MODEL_INFERENCE_OR_SETUP_FAILED")
        return result
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    meta = _probe(video)
    qc = inspect_clip(video)
    p = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video),
                        "-f", "null", "-"], capture_output=True, timeout=100)
    motion = gray_motion(video)
    checks = {
        "native_geometry": meta["width"] == 576 and meta["height"] == 1024,
        "native_frames": meta["frames"] == 33 and qc["decoded_frames"] == 33,
        "native_24fps": meta["fps"] == "24", "full_decode": p.returncode == 0,
        "no_neon": qc["status"] == "NO_NEON_ARTIFACT_FOUND",
        "receipt_sha": receipt.get("clip_sha256") == digest,
        "receipt_source_revision": receipt.get("studio_git_revision") == revision,
        "receipt_native_profile": receipt.get("pilot_profile") == PROFILE,
        "receipt_no_upscale": receipt.get("source_video_upscaled") is False,
        "receipt_correct_actor": receipt.get("shot") == SHOTS[alias],
    }
    result.update(mp4_sha256=digest, bytes=video.stat().st_size, probe=meta,
                  checks=checks, technical_pass=all(checks.values()),
                  color_status=qc["status"], motion_proxy=motion,
                  actual_inference_seconds=receipt.get("actual_gpu_seconds"))
    resource = output / "EB01_NATIVE_RESOURCES_SAFE.json"
    if resource.is_file():
        data = json.loads(resource.read_text())
        # Explicit allowlist: never publish arbitrary GPU-output JSON.
        result["resources"] = {k: data.get(k) for k in (
            "samples", "per_device", "host_ram_peak_gib", "process_rss_peak_gib",
            "cpu_percent_peak", "cpu_logical_count", "host_ram_total_gib",
            "host_ram_available_gib", "disk_free_gib", "wall_seconds", "finished")}
    return result


def run_pilot(alias: str, token: str, root: Path, run_id: str, revision: str,
              attempt: int, *, collect_run: str | None = None, poll_seconds: int = 2700) -> dict:
    report = {"schema": 1, "account_alias": alias, "pilot_profile": PROFILE,
              "gpu_submission_attempts": 0, "production_jobs_submitted": 0,
              "public_media_uploaded": False, "movie_ready": False,
              "owner_quality_approved": False, "run_id": collect_run or run_id}
    try:
        auth, handle = query(token)
        if auth != "AUTHENTICATED_READ_ONLY" or not handle:
            raise PilotBlocked("ACCOUNT_AUTH_NOT_VERIFIED")
        source_run = collect_run or run_id
        folder, kernel = prepare(root, alias, handle, source_run, revision)
        # A GitHub retry MUST use collection mode. A new generation requires
        # a new explicit trigger and its own run ID, never a retry loop.
        if attempt == 1 and collect_run is None:
            report["gpu_submission_attempts"] = 1
            call(token, ["kaggle", "kernels", "push", "-p", str(folder),
                         "--timeout", "3600", "--accelerator", "NvidiaTeslaT4"], timeout=220)
        deadline = time.monotonic() + poll_seconds
        while time.monotonic() < deadline:
            state_raw = call(token, ["kaggle", "kernels", "status", kernel], timeout=65)
            found = STATE.findall(state_raw)
            if len(found) != 1:
                raise PilotBlocked("UNKNOWN_KERNEL_STATE")
            state = found[0]
            report["kaggle_state"] = state
            print("EB01_NATIVE_PILOT_STATE", alias, state, flush=True)
            if state in ("COMPLETE", "ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED"):
                output = root / "output"
                output.mkdir(mode=0o700)
                call(token, ["kaggle", "kernels", "output", kernel,
                             "-p", str(output), "--force"], timeout=300)
                try:
                    verified = verify_output(output, alias, revision)
                except Exception as exc:
                    verified = {"actual_mp4_created": (output/"S02_PILOT33.mp4").is_file(),
                                "technical_pass": False, "error_type": type(exc).__name__,
                                "private_diagnostics_preserved": True}
                report["native_output"] = verified
                # Preserve original output even after a color/actor QA reject.
                # Only genuine MP4, small receipts and diagnostic log are packed.
                archive = root / "private-return.zip"
                selected = [p for p in output.iterdir() if p.is_file() and
                            p.name in {"S02_PILOT33.mp4", "EB01_S02_LTX_PILOT33_JOB.json",
                                       "EB01_S02_NATIVE_GPU_FRAME_QA_SAFE.json", "EB01_NATIVE_RESOURCES_SAFE.json",
                                       "EB01_NATIVE_CHECKPOINT_CONTEXT.json", "EB01_NATIVE_PREDECODE.safetensors",
                                       "EB01_LTX_PRIVATE_GPU_LOG.txt"}]
                with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
                    for path in selected:
                        z.write(path, path.name)
                context = {"purpose": "eb01-native-pilot", "run_id": source_run, "account": alias}
                seal(archive, Path("ops/eb01_native_pilot_return_public.pem"),
                     Path(os.environ["RUNNER_TEMP"]) / ("EB01_NATIVE_" + alias + ".twenc"), context)
                report["encrypted_private_return_created"] = True
                report["status"] = ("NATIVE_MP4_TECHNICAL_PASS_ARTISTIC_REVIEW_REQUIRED"
                                    if verified["technical_pass"] else "NATIVE_PILOT_FAILED_OR_QA_BLOCKED")
                return report
            time.sleep(40)
        report["status"] = "PILOT_STILL_QUEUED_OR_RUNNING_COLLECT_SAME_JOB_ONLY"
    except (PilotBlocked, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        report["status"] = "BLOCKED_" + (str(exc) if isinstance(exc, PilotBlocked) else type(exc).__name__)
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--account", choices=tuple(SHOTS), required=True)
    p.add_argument("--collect-run")
    args = p.parse_args()
    os.umask(0o077)
    with tempfile.TemporaryDirectory(dir=os.environ["RUNNER_TEMP"], prefix="eb01-native-private-") as t:
        report = run_pilot(args.account, os.environ.get("EB01_ACCOUNT_TOKEN", ""), Path(t),
                           os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"],
                           int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")), collect_run=args.collect_run)
    path = Path(os.environ["RUNNER_TEMP"]) / ("EB01_NATIVE_" + args.account + "_SAFE.json")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("EB01_NATIVE_PILOT_SAFE_RESULT", json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "NATIVE_MP4_TECHNICAL_PASS_ARTISTIC_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
