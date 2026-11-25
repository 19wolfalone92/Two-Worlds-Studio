"""Read-only forensic retrieval for the failed private recliner LTX kernel.
No GPU resubmission. Raw logs and recoverable native segments stay on the runner
and are sealed to the owner's recovery public key before artifact upload.
Only a redacted diagnostic summary is printed.
"""
from __future__ import annotations
import hashlib, json, os, re, subprocess, tempfile, zipfile
from pathlib import Path

KERNEL = os.environ.get("STUDIO_FORENSIC_KERNEL", "pavelbloody/studio-recliner-native-45480625216")
SOURCE_SHA256 = "9b1310c91f9718e29d026d6346dbc3ebd7687027a871ac5dbdf4a4f2e36895b3"
STAGE_RE = re.compile(
    r"\b(WAVE_STAGE_(?:BEGIN_MODEL_LOAD|END_MODEL_LOAD|BEGIN_PROMPT_ENCODE|"
    r"END_PROMPT_ENCODE|BEGIN_CONDITIONING_VAE|END_CONDITIONING_VAE|"
    r"BEGIN_TRANSFORMER|DENOISE_STEP_START|DENOISE_STEP_END|END_TRANSFORMER|"
    r"BEGIN_DECODE|END_DECODE|NATIVE_VIDEO_OUTPUT_READY))\b"
)
EXCEPTION_RE = re.compile(
    r"\b(SyntaxError|NameError|TypeError|ModuleNotFoundError|ImportError|"
    r"RuntimeError|ValueError|AssertionError|FileNotFoundError|OSError|"
    r"AttributeError|CUDAOutOfMemoryError|OutOfMemoryError|MemoryError|"
    r"TimeoutExpired|PermissionError|ConnectionError)\b"
)
SAFE_ERROR_RE = re.compile(
    r"(?i)(CUDA out of memory|out of memory|CUDNN_STATUS|CUDA error|"
    r"device-side assert|no space left|ModuleNotFoundError|ImportError|"
    r"RuntimeError|AssertionError|ValueError|TypeError|FileNotFoundError|"
    r"PermissionError|TimeoutExpired|timed out|SIGKILL|SIGTERM|Killed|"
    r"No such file or directory|invalid device|expected scalar type|"
    r"shape mismatch|size mismatch|not enough memory|failed to load|"
    r"cannot allocate|download failed|ConnectionError|NeuralBlocked|STUDIO_RECLINER_NATIVE_MODEL_BLOCKED|NVIDIA-SMI has failed)"
)
PRIVATE_LINE_RE = re.compile(
    r"(?i)(prompt|photorealistic|reference|STUDIO_RECLINER_PRIVATE_REFERENCE|"
    r"IMAGE_B64|CONFIG_B64|KAGGLE_API_TOKEN|Authorization|Bearer|"
    r"BEGIN PRIVATE KEY|data:image)"
)
FRAME_RE = re.compile(r'File "[^"]+", line \d+, in ([A-Za-z0-9_<>.]+)')
PATH_RE = re.compile(r"(?<![A-Za-z0-9])/(?:kaggle|root|tmp|home|working|input)/[^\s,;\"']+")
URL_RE = re.compile(r"https?://\S+")
LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_+/=-]{96,}\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_error_line(line: str) -> str | None:
    if PRIVATE_LINE_RE.search(line) or not SAFE_ERROR_RE.search(line):
        return None
    line = PATH_RE.sub("[PRIVATE_PATH]", line)
    line = URL_RE.sub("[REDACTED_URL]", line)
    line = LONG_TOKEN_RE.sub("[REDACTED_TOKEN_OR_PAYLOAD]", line)
    return line.strip()[-240:] or None


def summarize_logs(text: str) -> dict:
    stages = STAGE_RE.findall(text)
    exceptions = list(dict.fromkeys(EXCEPTION_RE.findall(text)))
    safe_errors = []
    for line in text.splitlines():
        safe = safe_error_line(line)
        if safe and safe not in safe_errors:
            safe_errors.append(safe)
    frames = FRAME_RE.findall(text)
    low = text.lower()
    return {
        "stages": list(dict.fromkeys(stages))[-40:],
        "last_stage": stages[-1] if stages else "NONE",
        "exception_classes": exceptions[-20:],
        "traceback_frame_lines_count": len(frames),
        "traceback_functions": frames[-20:],
        "safe_error_lines": safe_errors[-12:],
        "error_markers": {
            "cuda_oom": "cuda out of memory" in low or "outofmemoryerror" in low,
            "cuda_runtime": "cuda error" in low or "cudnn_status" in low,
            "missing_dependency": "modulenotfounderror" in low or "no module named" in low,
            "disk_full": "no space left on device" in low,
            "timeout": "timeoutexpired" in low or "timed out" in low,
            "worker_killed": "sigkill" in low or "sigterm" in low or "killed" in low,
            "missing_file": "no such file or directory" in low,
            "model_load": "failed to load" in low or "model load" in low,
        },
    }


def parse_state(text: str) -> str:
    matches = re.findall(r"KernelWorkerStatus\.([A-Z_]+)", text)
    return matches[-1] if matches else "UNKNOWN"


def _call(args: list[str], timeout: int = 180) -> tuple[int, str, str, str | None]:
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
        return result.returncode, result.stdout or "", result.stderr or "", None
    except subprocess.TimeoutExpired:
        return 124, "", "", "TimeoutExpired"
    except OSError as exc:
        return 125, "", "", type(exc).__name__


def _probe_video(path: Path) -> dict:
    result = {"file": path.name, "bytes": path.stat().st_size,
              "sha256": sha256_file(path), "ffprobe_ok": False, "full_decode_ok": False}
    probe = _call([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration",
        "-of", "json", str(path)
    ], timeout=60)
    if probe[0] == 0:
        try:
            streams = json.loads(probe[1]).get("streams", [])
            result["stream"] = streams[0] if streams else None
            result["ffprobe_ok"] = bool(streams)
        except (ValueError, TypeError):
            result["stream"] = None
    decode = _call(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"], timeout=120)
    result["full_decode_ok"] = decode[0] == 0
    if result["full_decode_ok"]:
        try:
            from tools.eb01_true_gpu_clip_numbers import gray_motion
            from studio.eb01_motion_artifact_gate import inspect_clip
            result["motion"] = gray_motion(path)
            qa = inspect_clip(path)
            result["color_gate"] = {
                "status": qa.get("status"),
                "decoded_frames": qa.get("decoded_frames"),
                "safe_frame_count": qa.get("safe_frame_count"),
                "flagged_frames": qa.get("flagged_frames", [])[:40],
                "spectral_intrusion_flagged_frames": qa.get("spectral_intrusion_flagged_frames", [])[:40],
                "horizontal_neon_band_flagged_frames": qa.get("horizontal_neon_band_flagged_frames", [])[:40],
                "neon_line_geometry_flagged_frames": qa.get("neon_line_geometry_flagged_frames", [])[:40],
            }
        except Exception as exc:
            result["quality_check_error_class"] = type(exc).__name__
    return result


def _is_recoverable_output(path: Path, relative: str) -> bool:
    name = path.name
    if name == "STUDIO_RECLINER_PRIVATE_REFERENCE.webp":
        return False
    if name.startswith((
        "STUDIO_RECLINER_REAL_SEGMENT_", "STUDIO_RECLINER_TRUE_NATIVE_",
        "STUDIO_RECLINER_PRIVATE_GPU_", "STUDIO_RECLINER_NATIVE_FAILURE",
    )):
        return True
    if name == "STUDIO_RECLINER_PRIVATE_LASTFRAME.png":
        return True
    if path.suffix.lower() == ".mp4" and "STUDIO_RECLINER_NATIVE_PRIVATE_GENERATION/segment_" in relative:
        return True
    if path.suffix.lower() == ".log" and "recliner" in name.lower():
        return True
    return False


def main() -> None:
    runner_temp = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir()))
    report_path = runner_temp / "STUDIO_RECLINER_FORENSIC_REPORT.json"
    sealed_path = runner_temp / "STUDIO_RECLINER_FORENSIC_RECOVERY.twenc"
    report = {
        "schema": 1, "kernel": KERNEL, "source_sha256": SOURCE_SHA256,
        "read_only": True, "new_gpu_submissions": 0, "new_gpu_inference": False,
        "public_private_photo_exposed": False, "published": False,
        "human_artistic_approved": False,
    }
    status_rc, status_out, status_err, status_exception = _call(
        ["kaggle", "kernels", "status", KERNEL], timeout=90
    )
    report["status_returncode"] = status_rc
    report["status_exception"] = status_exception
    report["kernel_state"] = parse_state(status_out + "\n" + status_err)

    with tempfile.TemporaryDirectory(prefix="studio-recliner-forensic-") as temp:
        output_dir = Path(temp)
        output_rc, _, _, output_exception = _call(
            ["kaggle", "kernels", "output", KERNEL, "-p", str(output_dir), "--force"],
            timeout=300,
        )
        report["output_download_returncode"] = output_rc
        report["output_download_exception"] = output_exception
        logs_rc, logs_out, logs_err, logs_exception = _call(
            ["kaggle", "kernels", "logs", KERNEL], timeout=180
        )
        report["logs_download_returncode"] = logs_rc
        report["logs_download_exception"] = logs_exception

        report["private_reference_file_detected_but_not_exported"] = any(
            p.is_file() and p.name == "STUDIO_RECLINER_PRIVATE_REFERENCE.webp"
            for p in output_dir.rglob("*")
        )
        log_parts = [logs_out, logs_err]
        for path in output_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".log":
                try:
                    log_parts.append(path.read_text(encoding="utf-8", errors="replace")[-100000:])
                except OSError:
                    pass
        combined = "\n".join(log_parts)[-250000:]
        report["logs_sha256"] = hashlib.sha256(combined.encode("utf-8", errors="replace")).hexdigest()
        report["logs_bytes_observed"] = len(combined.encode("utf-8", errors="replace"))
        report.update(summarize_logs(combined))

        files, videos, recoverable = [], [], []
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file() or path.name == "STUDIO_RECLINER_PRIVATE_REFERENCE.webp":
                continue
            relative = path.relative_to(output_dir).as_posix()
            files.append({"file": relative[:180], "bytes": path.stat().st_size})
            if path.suffix.lower() == ".mp4" and (
                path.name.startswith(("STUDIO_RECLINER_REAL_SEGMENT_", "STUDIO_RECLINER_TRUE_NATIVE_"))
                or "STUDIO_RECLINER_NATIVE_PRIVATE_GENERATION/segment_" in relative
            ):
                videos.append(_probe_video(path))
            if _is_recoverable_output(path, relative):
                recoverable.append((path, relative))

        report["output_file_count_excluding_private_reference"] = len(files)
        report["output_files"] = files[:80]
        report["native_video_candidates"] = videos[:20]

        failure_receipt = output_dir / "STUDIO_RECLINER_TRUE_NATIVE_FAILURE_SAFE.json"
        if failure_receipt.is_file():
            try:
                value = json.loads(failure_receipt.read_text(encoding="utf-8"))
                report["failure_receipt"] = {
                    key: value.get(key) for key in ("stage", "error_type", "successful_native_segments")
                }
            except (OSError, ValueError, TypeError):
                report["failure_receipt"] = {"valid_json": False}
        receipt = output_dir / "STUDIO_RECLINER_TRUE_NATIVE_RECEIPT.json"
        if receipt.is_file():
            try:
                value = json.loads(receipt.read_text(encoding="utf-8"))
                report["native_receipt"] = {
                    key: value.get(key) for key in (
                        "status", "model", "output_frames", "output_width", "output_height",
                        "output_fps", "actual_native_gpu_inference"
                    )
                }
            except (OSError, ValueError, TypeError):
                report["native_receipt"] = {"valid_json": False}

        report["recoverable_file_count"] = len(recoverable)
        if recoverable:
            archive_path = runner_temp / "STUDIO_RECLINER_FORENSIC_RECOVERY.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                for path, relative in recoverable:
                    archive.write(path, arcname=relative)
            try:
                from studio.sealed_artifacts import seal
                seal(
                    archive_path,
                    Path("ops/studio_recliner_owner_recovery_public.pem"),
                    sealed_path,
                    {
                        "purpose": "studio-recliner-forensic-recovery",
                        "schema": 1, "kernel": KERNEL, "source_sha256": SOURCE_SHA256,
                    },
                )
                report["recovery_bundle"] = {
                    "sealed": True, "bytes": sealed_path.stat().st_size,
                    "sha256": sha256_file(sealed_path), "raw_media_uploaded": False,
                }
            except Exception as exc:
                report["recovery_bundle"] = {
                    "sealed": False, "error_class": type(exc).__name__, "raw_media_uploaded": False,
                }
            finally:
                archive_path.unlink(missing_ok=True)
        else:
            report["recovery_bundle"] = {
                "sealed": False, "reason": "NO_RECOVERABLE_OUTPUT_FILES", "raw_media_uploaded": False,
            }

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("STUDIO_RECLINER_PRIVATE_GPU_FORENSIC", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
