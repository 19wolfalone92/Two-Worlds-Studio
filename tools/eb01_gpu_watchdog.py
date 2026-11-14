"""Autonomous read-only EB01 Kaggle monitor with root-cause triage.

Important: NEVER submits/restarts GPU jobs, and NEVER publishes media.
The GH workflow re-invokes it; a QUEUED Kaggle kernel is resource waiting,
not a FAILED video model. Genuine completed MP4 requires a private receipt,
exact SHA256, 33 decoded frames, FFmpeg and visual artifact scan, then it
is still only an UNAPPROVED SHORT PILOT, not a 40s film.

Python standard library for status and diagnosis; numpy needed only if
a private MP4 actually exists.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

KERNEL = "wolf19loner92/eb01-ltx2b-pilot33-fd6d11a64bd6"
EXPECTED_JOB = "EB01_S02_LTX_PILOT33_JOB.json"
EXPECTED_MP4 = "S02_PILOT33.mp4"
KNOWN = ("QUEUED", "RUNNING", "COMPLETE", "ERROR", "CANCEL")
MAX_OUTPUT_BYTES = 500_000_000


def parse_state(raw: str) -> str:
    hits = re.findall(r"KernelWorkerStatus\.(QUEUED|RUNNING|COMPLETE|ERROR|CANCEL)", raw)
    if not hits:
        return "UNKNOWN"
    return hits[-1]


def classify_failure(text: str, receipt: dict | None) -> str:
    evidence = (text + "\n" + str((receipt or {}).get("error_summary", ""))).lower()
    if "timeout" in evidence or "timed out" in evidence:
        if "float16" in evidence and "cpu" in evidence:
            return "INFERENCE_TIMEOUT_WITH_UNSUPPORTED_FP16_CPU_PATH"
        return "INFERENCE_TIMEOUT_NEEDS_STAGE_TIMING"
    if "cuda out of memory" in evidence or "outofmemoryerror" in evidence:
        return "CUDA_VRAM_EXHAUSTION"
    if "float16" in evidence and "cpu" in evidence:
        return "UNSUPPORTED_CPU_HALF_PRECISION_VAE_OR_ENCODER"
    if "repository not found" in evidence or "401 unauthorized" in evidence:
        return "PRIVATE_SOURCE_ACCESS_FAILURE"
    if "no module named" in evidence or "importerror" in evidence:
        return "MISSING_DEPENDENCY"
    if "disk quota" in evidence or "no space left" in evidence:
        return "DISK_RESOURCE_FAILURE"
    if "connection" in evidence and ("timed out" in evidence or "refused" in evidence):
        return "EXTERNAL_MODEL_DOWNLOAD_NETWORK"
    return "UNCLASSIFIED_REAL_KERNEL_ERROR_NEEDS_TRACEBACK"


def sanitize(text: str) -> list[str]:
    """Never expose voice tokens, private paths, full original prompts or secrets."""
    result = []
    for line in text.splitlines():
        if not re.search(
            r"(?i)(error|traceback|exception|out of memory|oom|cuda|float16|failed|timeout|timed out|importerror|no module|queue|workerstatus)",
            line,
        ):
            continue
        line = re.sub(r"hf_[A-Za-z0-9]{12,}|sk-[A-Za-z0-9_-]{12,}", "[TOKEN_REDACTED]", line)
        line = re.sub(
            r"(?i)(authorization|kaggle_api_token|api_key|password)\s*[:=]\s*\S+",
            r"\1=[REDACTED]",
            line,
        )
        line = re.sub(r"/kaggle/input/\S+", "[PRIVATE_INPUT]", line)
        line = re.sub(r"(?i)prompt\s*[:=]\s*.+", "prompt=[PRIVATE]", line)
        result.append(line[:220])
    return result[-18:]


def run_cli(args: list[str], timeout: int = 90) -> tuple[int, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + "\n" + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "CLI_TIMEOUT"
    except OSError as e:
        return 127, "CLIENT_UNAVAILABLE " + type(e).__name__


def file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as fp:
        for buf in iter(lambda: fp.read(1024*1024), b""):
            result.update(buf)
    return result.hexdigest()


def validate_actual_pilot(root: Path) -> dict:
    """No real MP4 is accepted from job-status text or a fabricated receipt."""
    file = root / EXPECTED_MP4
    receipt = root / EXPECTED_JOB
    if not (file.is_file() and receipt.is_file()):
        raise ValueError("Actual MP4 or original GPU receipt missing")
    if not (1024 <= file.stat().st_size <= MAX_OUTPUT_BYTES):
        raise ValueError("GPU clip is empty or oversize")
    data = json.loads(receipt.read_text(encoding="utf-8"))
    if (data.get("schema") != 7 or data.get("shot") != "S02"
        or data.get("success") is not True or data.get("mode") != "pilot"
        or data.get("frame_count") != 33 or data.get("fps") != 24
        or data.get("width") != 320 or data.get("height") != 576
        or data.get("whole_movie_ready") is not False
        or data.get("human_artistic_approved") is not False
        or data.get("published") is not False
        or data.get("audio_conditioned_lipsync_verified") is not False
        or data.get("original_prompt_passed_unchanged") is not True):
        raise ValueError("Original model inference receipt or private movie status invalid")
    sha = file_sha(file)
    if sha != data.get("clip_sha256"):
        raise ValueError("Original private GPU MP4 SHA256 mismatch")
    info = subprocess.check_output(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,width,height,avg_frame_rate",
         "-of", "json", str(file)],
        text=True, timeout=60)
    stream = json.loads(info)["streams"][0]
    if (stream["nb_read_frames"] != "33" or stream["avg_frame_rate"] != "24/1"
        or stream["width"] != 320 or stream["height"] != 576):
        raise ValueError("Real FFmpeg decode contradicts native source frame receipt")
    from studio.eb01_motion_artifact_gate import inspect_clip
    qa = inspect_clip(file)
    if qa.get("decoded_frames") != 33:
        raise ValueError("Full-frame source inspection incomplete")
    # Even with no detected RGB artifact, a human must evaluate bone anatomy,
    # natural head/jaw/shoulder movement, and face/character consistency.
    return {
        "original_pilot_sha256": sha,
        "decoded_true_neural_frames": qa["decoded_frames"],
        "artifact_scan_status": qa["status"],
        "narrow_neon_suspected_frames": qa["narrow_neon_suspected_frames"],
        "full_human_artistic_approval": False,
        "full_episode_ready": False,
        "source_stage": "TRUE_PRIVATE_GPU_NATIVE_I2V_UNAPPROVED_PILOT",
    }


def inspect(kernel: str, root: Path, *, max_polls: int = 4,
            poll_seconds: int = 30, runner=run_cli, sleeper=time.sleep) -> dict:
    """Poll existing private GPU job. No auto retry or new kernel creation."""
    if kernel != KERNEL:
        raise ValueError("Only previously submitted locked private kernel permitted")
    root.mkdir(parents=True, exist_ok=True)
    if not (1 <= max_polls <= 20 and 0 <= poll_seconds <= 120):
        raise ValueError("Bounded monitoring interval required")
    report = {
        "schema": 1, "kernel": kernel,
        "monitor_read_only": True, "gpu_job_submitted": False,
        "paid_generation_used": False, "published": False,
        "source_branch": "fix/eb01-s02-private-raw-preservation-20261113",
        "full_movie_ready": False,
    }
    history = []
    for i in range(max_polls):
        code, body = runner(["kaggle", "kernels", "status", kernel], timeout=55)
        state = parse_state(body) if code == 0 else "UNKNOWN"
        history.append(state)
        report["kaggle_state"] = state
        # QUEUED is NOT a failure and should not occupy a paid GH runner
        # for 45 minutes. Let next hourly monitor observe the same kernel.
        if state == "QUEUED":
            report["status"] = "WAITING_FOR_FREE_KAGGLE_GPU"
            report["next_action"] = "MONITOR_EXISTING_KERNEL_NO_RESUBMISSION"
            break
        if state == "RUNNING":
            report["status"] = "INFERENCE_IN_PROGRESS_NO_RESULT_YET"
            report["next_action"] = "MONITOR_EXISTING_KERNEL_NO_RESUBMISSION"
            if i + 1 < max_polls:
                sleeper(poll_seconds)
                continue
            break
        if state in ("ERROR", "CANCEL", "COMPLETE"):
            out = root / "private_download"
            out.mkdir(exist_ok=True)
            exit_code, summary = runner(
                ["kaggle", "kernels", "output", kernel, "-p", str(out), "--force"],
                timeout=240)
            report["download_success"] = exit_code == 0
            if state == "COMPLETE":
                try:
                    if exit_code != 0:
                        raise ValueError("Kaggle kernel output download failed")
                    report["clip_verification"] = validate_actual_pilot(out)
                    if report["clip_verification"]["artifact_scan_status"] == "NO_NEON_ARTIFACT_FOUND":
                        report["status"] = "TRUE_PILOT_TECHNICAL_QA_PASS_HUMAN_REVIEW_REQUIRED"
                    else:
                        report["status"] = "TRUE_PILOT_VISUAL_COLOR_ARTIFACT_BLOCKED"
                    report["next_action"] = "HUMAN_INSPECT_ALL_33_FRAMES"
                except (ValueError, OSError, KeyError, subprocess.SubprocessError) as ex:
                    report["status"] = "KERNEL_COMPLETE_BUT_NO_VALID_TRUE_VIDEO"
                    report["failure_category"] = type(ex).__name__
                    report["next_action"] = "FIX_REAL_VIDEO_OR_RECEIPT"
                break
            receipt_file = out / EXPECTED_JOB
            receipt = None
            if receipt_file.is_file():
                try:
                    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
                except (ValueError, UnicodeDecodeError):
                    pass
            _, logs = runner(["kaggle", "kernels", "logs", kernel], timeout=70)
            disk_log = out / "EB01_LTX_PRIVATE_GPU_LOG.txt"
            disk_text = disk_log.read_text(errors="replace")[-10000:] if disk_log.is_file() else ""
            evidence = (disk_text + "\n" + logs)[-25000:]
            report["status"] = "REAL_GPU_INFERENCE_ERROR_DIAGNOSED"
            report["failure_category"] = classify_failure(evidence, receipt)
            report["diagnostic_lines_redacted"] = sanitize(evidence)
            report["gpu_receipt_success"] = (receipt or {}).get("success")
            report["next_action"] = "ROOT_CAUSE_PATCH_AND_TEST_FIRST_NO_AUTORETRY"
            break
        report["status"] = "KAGGLE_STATUS_UNREADABLE"
        report["next_action"] = "FIX_AUTH_OR_CLIENT_NO_GPU_RESUBMISSION"
        break
    report["observations"] = history
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Read-only private Kaggle S02 GPU watchdog")
    p.add_argument("--kernel", default=KERNEL)
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--max-polls", type=int, default=4)
    p.add_argument("--poll-seconds", type=int, default=30)
    a = p.parse_args(argv)
    try:
        result = inspect(a.kernel, a.folder, max_polls=a.max_polls,
                         poll_seconds=a.poll_seconds)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        result = {"schema":1, "status":"WATCHDOG_INFRA_ERROR",
                  "failure_category":type(exc).__name__,
                  "monitor_read_only":True,"published":False,
                  "full_movie_ready":False,"gpu_job_submitted":False}
    a.folder.mkdir(parents=True, exist_ok=True)
    (a.folder/"EB01_S02_EXISTING_KERNEL_WATCHDOG_REPORT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    print("EB01_KAGGLE_GPU_CONTROL",json.dumps({
        k:v for k,v in result.items() if k in
        ("kernel","kaggle_state","status","failure_category","next_action",
         "observations","full_movie_ready","published")
    },ensure_ascii=False),flush=True)
    # Never mark ERROR as a successful video production.
    return 2 if result["status"] in (
        "REAL_GPU_INFERENCE_ERROR_DIAGNOSED",
        "KERNEL_COMPLETE_BUT_NO_VALID_TRUE_VIDEO",
        "TRUE_PILOT_VISUAL_COLOR_ARTIFACT_BLOCKED",
        "KAGGLE_STATUS_UNREADABLE","WATCHDOG_INFRA_ERROR"
    ) else 0


if __name__=="__main__":
    sys.exit(main())
