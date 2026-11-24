"""Bounded original native LTX model subprocess with real stage telemetry.

No model output is broadcast to a public GitHub log. Only strict stage label
tokens can be printed. No automatic retries or GPU over-budget fallbacks.
"""
from __future__ import annotations
import os
import re
import signal
import subprocess
import time
from pathlib import Path

_SUPPORTED_STAGE_PREFIXES = frozenset({"EB01", "WAVE"})
def _stage_regex(stage_prefix: str) -> re.Pattern[str]:
    # Reject arbitrary prefixes before invoking any GPU process.
    if stage_prefix not in _SUPPORTED_STAGE_PREFIXES:
        raise ValueError("Unsupported native inference stage prefix")
    return re.compile(r"(?m)^" + re.escape(stage_prefix) +
                      r"_STAGE_([A-Z0-9_]{2,64})(?:\s|$)")
_STAGE_ALLOWED = frozenset({
    "BEGIN_MODEL_LOAD","END_MODEL_LOAD","BEGIN_PROMPT_ENCODE",
    "END_PROMPT_ENCODE","BEGIN_CONDITIONING_VAE",
    "END_CONDITIONING_VAE","BEGIN_TRANSFORMER",
    "DENOISE_STEP_START","DENOISE_STEP_END",
    "END_TRANSFORMER","BEGIN_DECODE","END_DECODE",
    "NATIVE_VIDEO_OUTPUT_READY",
})

class InferenceTimeBudgetExpired(RuntimeError):
    def __init__(self,elapsed_seconds: float,last_stage: str):
        super().__init__(
            f"Native inference exceeded bounded GPU budget at {last_stage}"
        )
        self.elapsed_seconds=elapsed_seconds
        self.last_stage=last_stage

def latest_stage(path: Path, *, stage_prefix: str = "EB01") -> str:
    matcher = _stage_regex(stage_prefix)
    if not path.is_file():
        return "NO_NATIVE_STAGE_MARKER"
    with path.open("rb") as fp:
        fp.seek(0,2)
        fp.seek(max(0,fp.tell()-65536))
        tail=fp.read().decode("utf-8",errors="replace")
    matches=[m.group(1) for m in matcher.finditer(tail)
             if m.group(1) in _STAGE_ALLOWED]
    return matches[-1] if matches else "NO_NATIVE_STAGE_MARKER"

def _stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=8)

def run_native_bounded(
    argv: list[str], *,
    cwd: Path,
    env: dict[str,str],
    log_path: Path,
    deadline_seconds: float = 1400,
    heartbeat_seconds: float = 20,
    monotonic=time.monotonic,
    stage_prefix: str = "EB01",
) -> dict:
    if (not isinstance(argv,list) or not argv or
        not 10 <= deadline_seconds <= 3600 or
        not 1 <= heartbeat_seconds <= 120):
        raise ValueError("Only bounded, non-shell original model invocation allowed")
    _stage_regex(stage_prefix)  # Validate before starting child.
    if not Path(cwd).is_dir():
        raise ValueError("Model source working directory absent")
    log_path.parent.mkdir(parents=True,exist_ok=True)
    start=monotonic()
    last="NO_NATIVE_STAGE_MARKER"
    with log_path.open("wb") as log:
        process=subprocess.Popen(
            argv,cwd=str(cwd),env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            close_fds=True)
        try:
            while process.poll() is None:
                remaining=deadline_seconds-(monotonic()-start)
                if remaining <= 0:
                    stage=latest_stage(log_path,stage_prefix=stage_prefix)
                    raise InferenceTimeBudgetExpired(
                        round(monotonic()-start,1),stage)
                try:
                    process.wait(timeout=min(heartbeat_seconds,remaining))
                except subprocess.TimeoutExpired:
                    stage=latest_stage(log_path,stage_prefix=stage_prefix)
                    if stage != last:
                        print("EB01_GPU_STAGE_HEARTBEAT",
                              "stage="+stage,
                              "seconds="+str(round(monotonic()-start,1)),
                              flush=True)
                        last=stage
            elapsed=round(monotonic()-start,2)
            stage=latest_stage(log_path,stage_prefix=stage_prefix)
            if process.returncode:
                raise RuntimeError(
                    "Authentic neural pipeline subprocess failed"
                    f" exit={process.returncode} last_stage={stage}"
                )
            if stage!="NATIVE_VIDEO_OUTPUT_READY":
                raise RuntimeError(
                    "Subprocess returned 0 without proved native video completion"
                    f" last_stage={stage}"
                )
            return {
                "returncode":0,"elapsed_seconds":elapsed,
                "last_stage":stage,
                "native_pipeline_completed":True,
                "log_sha256":None,
            }
        except BaseException:
            _stop_process_group(process)
            raise
