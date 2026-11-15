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

KERNEL = "wolf19loner92/eb01-ltx2b-pilot33-c28a516c464e"
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
    # Prefer the actual terminal GPU receipt over benign library warnings.
    # Incident 2026-11-14: real T4 generated a MP4 in ~330s but the
    # frame/color QA rejected it. A warning about CPU/FP16 was emitted
    # earlier during safe weight offloading; it was NOT the fatal error.
    terminal = str((receipt or {}).get("error_summary", "")).lower()
    if "video frame/color qa" in terminal or "frame/color qa" in terminal:
        return "GENUINE_GPU_VIDEO_GENERATED_BUT_VISUAL_COLOR_OR_GEOMETRY_QA_FAILED"
    if "no unique actual generated video" in terminal:
        return "NEURAL_INFERENCE_FINISHED_NO_UNIQUE_MP4"
    if "pilot inference exceeded" in terminal or "model stalled at" in terminal:
        return "MODEL_INFERENCE_BUDGET_EXCEEDED"
    evidence = (text + "\n" + terminal).lower()
    if "video frame/color qa" in evidence:
        return "GENUINE_GPU_VIDEO_GENERATED_BUT_VISUAL_COLOR_OR_GEOMETRY_QA_FAILED"
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


def private_numeric_qa_summary(root: Path) -> dict:
    """Keep only the ORIGINAL GPU frame-clock + anomaly measurements.

    Reconstruct from the real MP4 if the GPU worker was old and had no QA
    receipt. Never include source portrait bytes, audio, prompt or raw log.
    """
    saved = root / "EB01_S02_NATIVE_GPU_FRAME_QA_SAFE.json"
    fields = (
        "rgb_gate_status", "frame_flags", "spectral_flags",
        "horizontal_band_flags", "narrow_neon_flags", "checks",
        "actual_native_gpu_generated", "actual_gpu_seconds"
    )
    if saved.is_file():
        try:
            data=json.loads(saved.read_text(encoding="utf-8"))
            if data.get("schema")==1 and data.get("shot")=="S02":
                return {"source":"PRIVATE_WORKER_ORIGINAL_QA_RECEIPT",
                        **{k:data.get(k) for k in fields if k in data}}
        except (ValueError,UnicodeError):
            return {"source":"INVALID_GPU_QA_RECEIPT"}
    video=root/"S02_PILOT33.mp4"
    if video.is_file():
        try:
            from studio.eb01_motion_artifact_gate import inspect_clip
            from studio.comfy_neural import _probe
            meta=_probe(video)
            qa=inspect_clip(video)
            return {
                "source":"RECONSTRUCTED_FROM_ACTUAL_PRIVATE_MP4",
                "native_frame_count":meta["frames"],
                "fps":meta["fps"],"width":meta["width"],"height":meta["height"],
                "rgb_gate_status":qa["status"],
                "frame_flags":qa.get("flagged_frames",[])[:100],
                "spectral_flags":qa.get("spectral_intrusion_flagged_frames",[])[:100],
                "horizontal_band_flags":qa.get("horizontal_neon_band_flagged_frames",[])[:100],
                "narrow_neon_flags":qa.get("narrow_neon_suspected_frames",[])[:100],
                "original_video_gpu_generated":True,
            }
        except Exception as exc:
            return {"source":"PRIVATE_VIDEO_QA_RECONSTRUCTION_FAILED",
                    "failure_class":type(exc).__name__}
    return {"source":"PRIVATE_GPU_OUTPUT_HAS_NO_NATIVE_VIDEO_OR_QA_RECEIPT"}


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
    # Reuse the already-successful, PRIVATE Kaggle readback. A second
    # downloader can intermittently report 404 despite this exact worker
    # having obtained and SHA-verified the same MP4. Analyze actual decoded
    # motion here BEFORE the runner securely erases media.
    from tools.eb01_true_gpu_clip_numbers import gray_motion
    motion = gray_motion(file)
    # Motion is merely an optical proxy, NEVER proof of acting, identity,
    # facial anatomy, jaw/lipsync or owner approval. Keep every source private.
    # Even with no detected RGB artifact, a human must evaluate bone anatomy,
    # natural head/jaw/shoulder movement, and face/character consistency.
    return {
        "original_pilot_sha256": sha,
        "decoded_true_neural_frames": qa["decoded_frames"],
        "native_luma_motion_proxy": motion,
        "artifact_scan_status": qa["status"],
        "narrow_neon_suspected_frames": qa["narrow_neon_suspected_frames"],
        "full_human_artistic_approval": False,
        "full_episode_ready": False,
        "source_stage": "TRUE_PRIVATE_GPU_NATIVE_I2V_UNAPPROVED_PILOT",
    }



def validate_actual_qa_blocked_pilot(root: Path) -> dict:
    """A true decoded neural MP4 can exist even when independent QA fails.

    A completed MODEL is not an approved CLIP or finished 40s film.
    Check SHA and all 33 frames independently; fail closed on forged receipts.
    """
    movie=root/EXPECTED_MP4
    record=root/EXPECTED_JOB
    if not (movie.is_file() and record.is_file()
            and 1024<=movie.stat().st_size<=MAX_OUTPUT_BYTES):
        raise ValueError("Original review-blocked native MP4/receipt absent")
    receipt=json.loads(record.read_text(encoding="utf-8"))
    if (receipt.get("schema")!=7 or receipt.get("shot")!="S02"
        or receipt.get("status")!="REAL_NATIVE_GPU_VIDEO_GENERATED_QA_REVIEW_BLOCKED"
        or receipt.get("success") is not False
        or receipt.get("model_inference_completed") is not True
        or receipt.get("true_gpu_neural_inference") is not True
        or receipt.get("whole_movie_ready") is not False
        or receipt.get("human_artistic_approved") is not False
        or receipt.get("published") is not False
        or receipt.get("frame_count")!=33
        or str(receipt.get("fps")) not in ("24","24/1")
        or receipt.get("width")!=320 or receipt.get("height")!=576
        or receipt.get("clip_sha256")!=file_sha(movie)):
        raise ValueError("Unapproved native GPU film receipt failed proof")
    from studio.comfy_neural import _probe
    probe=_probe(movie)
    if (probe["frames"]!=33 or probe["width"]!=320 or
        probe["height"]!=576 or probe["fps"]!="24"):
        raise ValueError("Video bytes disagree with original native frame receipt")
    from studio.eb01_motion_artifact_gate import inspect_clip
    qa=inspect_clip(movie)
    if (qa["decoded_frames"]!=33 or qa["status"]=="NO_NEON_ARTIFACT_FOUND"):
        raise ValueError("Review-blocked receipt inconsistent with fresh color scan")
    return {
        "actual_original_video_sha256":receipt["clip_sha256"],
        "actual_neural_frames_verified":33,
        "native_video_fps":24,
        "frame_color_review_status":qa["status"],
        "review_color_flags":qa.get("flagged_frames",[])[:33],
        "spatial_line_geometry_flags":qa.get("neon_line_geometry_flagged_frames",[])[:33],
        "actor_artistic_approved":False,
        "full_movie_ready":False,
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
                    receipt_path=out/EXPECTED_JOB
                    candidate=(json.loads(receipt_path.read_text(encoding="utf-8"))
                               if receipt_path.is_file() else {})
                    if candidate.get("status")=="REAL_NATIVE_GPU_VIDEO_GENERATED_QA_REVIEW_BLOCKED":
                        report["clip_verification"]=validate_actual_qa_blocked_pilot(out)
                        report["status"]="REAL_GPU_NATIVE_33F_QA_REVIEW_BLOCKED_NOT_MODEL_CRASH"
                        report["failure_category"]="ACTUAL_INDEPENDENT_VISUAL_QA_REQUIRES_OWNER_REVIEW"
                        report["next_action"]="REVIEW_PRIVATE_NATIVE_MP4_NO_BLIND_GPU_RESTART"
                        break
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
            # Future failures report ACTUAL independent frame reasons rather
            # than collapsing a visual-QA rejection into a benign FP16 warning.
            report["real_native_video_qa"] = private_numeric_qa_summary(out)
            from tools.eb01_bounded_inference import latest_stage
            report["last_real_gpu_stage"] = latest_stage(disk_log)

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
        "REAL_GPU_NATIVE_33F_QA_REVIEW_BLOCKED_NOT_MODEL_CRASH",
        "KAGGLE_STATUS_UNREADABLE","WATCHDOG_INFRA_ERROR"
    ) else 0


if __name__=="__main__":
    sys.exit(main())
