"""Resume the SAME private Studio wave kernel from its verified 33-frame LTX result.

CPU-only preparation. Does not submit Kaggle jobs. All original user media and
private scripts remain in runner temp or Kaggle private storage. The returned
checkpoint is sealed for its owner before any overwrite of the original job.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from studio.sealed_artifacts import seal
from tools.studio_wave_status_gate import KERNEL, SOURCE_SHA256, parse_state

SEGMENT_SHA = "0e2f9c33e172ef48d8bf1761c06768eea661387b954830430071a28074803885"
BRANCH = "feature/studio-native-man-wave-ltx-20261123"
NEURAL_OUTPUT_SUBFOLDER = "STUDIO_WAVE_NATIVE_PRIVATE_GENERATION/segment_0"
SEGMENT_NAME = "video_output_0_photorealistic-lockedoff_2026112301_1024x576x33_0.mp4"


def command(args: list[str], timeout: int = 240) -> str:
    """Never expose Kaggle's raw error, private paths, or bearer tokens."""
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("STUDIO_SAFE_KAGGLE_COMMAND_UNAVAILABLE") from exc
    if p.returncode:
        raise RuntimeError("STUDIO_SAFE_KAGGLE_COMMAND_FAILED")
    return p.stdout


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stage_elapsed_seconds(public_safe_log: str) -> float:
    """Extract actual per-event timestamps, never manufacture inference time."""
    events = []
    try:
        decoded = json.loads(public_safe_log)
        if isinstance(decoded, list):
            events.extend(x for x in decoded if isinstance(x, dict))
        elif isinstance(decoded, dict):
            events.extend(decoded.get("logs", []))
    except (ValueError, TypeError):
        pass
    for line in public_safe_log.splitlines():
        part = line.strip().lstrip(",").rstrip(",")
        if not part.startswith("{"):
            continue
        try:
            item = json.loads(part)
            if isinstance(item, dict):
                events.append(item)
        except ValueError:
            continue
    begin, end = [], []
    for event in events:
        payload = event.get("data", "")
        time = event.get("time")
        if not isinstance(payload, str) or not isinstance(time, (int, float)):
            continue
        if "STUDIO_WAVE_NATIVE_LTX_SAMPLE_START" in payload:
            begin.append(float(time))
        if "WAVE_STAGE_NATIVE_VIDEO_OUTPUT_READY" in payload:
            end.append(float(time))
    if not begin or not end:
        raise RuntimeError("STUDIO_CHECKPOINT_NEURAL_TIMING_NOT_PROVEN")
    elapsed = end[0] - begin[0]
    if not 0 < elapsed < 1400:
        raise RuntimeError("STUDIO_CHECKPOINT_NEURAL_TIMING_IMPLAUSIBLE")
    return round(elapsed, 3)


def private_source_reference(private_kernel_code: str) -> bytes:
    """Only extract the original JPEG from Kaggle's authenticated private pull."""
    matches = re.findall(r'^IMAGE_B64="([A-Za-z0-9+/=]+)"$', private_kernel_code, re.M)
    if len(matches) != 1:
        raise RuntimeError("STUDIO_AMBIGUOUS_PRIVATE_ORIGINAL_REFERENCE")
    photo = base64.b64decode(matches[0], validate=True)
    if hashlib.sha256(photo).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("STUDIO_PRIVATE_REFERENCE_HASH_MISMATCH")
    return photo


def prepare(root: Path, *, revision: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise RuntimeError("STUDIO_REPAIR_SOURCE_COMMIT_NOT_PINNED")
    if parse_state(command(["kaggle", "kernels", "status", KERNEL])) != "ERROR":
        raise RuntimeError("STUDIO_NEVER_OVERWRITE_NONTERMINAL_GPU_KERNEL")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    old = root / "old_private_output"
    old.mkdir(mode=0o700)
    command(["kaggle", "kernels", "output", KERNEL,
             "-p", str(old), "--force"], timeout=300)
    clip = old / NEURAL_OUTPUT_SUBFOLDER / SEGMENT_NAME
    if not clip.is_file() or sha(clip) != SEGMENT_SHA:
        raise RuntimeError("STUDIO_ORIGINAL_MODEL_CHECKPOINT_ABSENT_OR_CHANGED")
    if len(list(old.rglob("*.mp4"))) != 1:
        raise RuntimeError("STUDIO_UNEXPECTED_AMBIGUOUS_PREVIOUS_NEURAL_OUTPUT")
    # The previous private model log proves that the subprocess reached
    # the real pinned-upstream LTX mp4 muxer before the outer guard failed.
    raw_model_log = old / "STUDIO_WAVE_PRIVATE_GPU_0.log"
    if not raw_model_log.is_file() or "WAVE_STAGE_NATIVE_VIDEO_OUTPUT_READY" not in raw_model_log.read_text(errors="replace"):
        raise RuntimeError("STUDIO_ORIGINAL_NATIVE_VIDEO_UPSTREAM_STAGE_NOT_PROVED")
    from studio.eb01_motion_artifact_gate import inspect_clip
    from tools.eb01_true_gpu_clip_numbers import gray_motion
    from studio.comfy_neural import _probe
    meta = _probe(clip)
    if any((meta.get("frames") != 33, meta.get("fps") != "24",
            meta.get("width") != 576, meta.get("height") != 1024)):
        raise RuntimeError("STUDIO_ORIGINAL_NEURAL_CHECKPOINT_GEOMETRY_MISMATCH")
    command(["ffmpeg", "-v", "error", "-xerror", "-i", str(clip),
             "-f", "null", "-"], timeout=90)
    motion = gray_motion(clip)
    color = inspect_clip(clip)
    if (color.get("status") != "NO_NEON_ARTIFACT_FOUND"
            or motion.get("decoded_native_frames") != 33
            or motion.get("mean_frame_luma_delta", 0) <= 0.18
            or motion.get("nonidentical_motion_frames", 0) < 4):
        raise RuntimeError("STUDIO_REAL_CHECKPOINT_FAILED_INDEPENDENT_NUMERIC_QUALITY")
    measured = stage_elapsed_seconds(
        command(["kaggle", "kernels", "logs", KERNEL], timeout=150))
    old_kernel = root / "old_private_kernel"
    old_kernel.mkdir(mode=0o700)
    command(["kaggle", "kernels", "pull", KERNEL,
             "-p", str(old_kernel), "-m"], timeout=240)
    candidates = list(old_kernel.glob("*.py"))
    if len(candidates) != 1:
        raise RuntimeError("STUDIO_AMBIGUOUS_PRIVATE_KERNEL_SOURCE")
    jpeg = private_source_reference(candidates[0].read_text())
    original_template = Path("tools/kaggle_studio_man_wave_native.py").read_text()
    config = {
        "schema": 1, "mode": "studio-native-wave",
        "private": True, "no_paid_services": True,
        "allow_social_publish": False, "human_artistic_approved": False,
        "branch": BRANCH, "revision": revision, "source_sha256": SOURCE_SHA256,
        "checkpoint_0_sha256": SEGMENT_SHA,
        "checkpoint_0_model_seconds": measured,
    }
    substitutions = [
        ("__STUDIO_WAVE_CONFIG_B64__", base64.b64encode(
            json.dumps(config, sort_keys=True).encode()).decode()),
        ("__STUDIO_WAVE_IMAGE_B64__", base64.b64encode(jpeg).decode()),
        ("__STUDIO_WAVE_CHECKPOINT_B64__", base64.b64encode(clip.read_bytes()).decode()),
    ]
    for needle, repl in substitutions:
        if original_template.count(needle) != 1:
            raise RuntimeError("STUDIO_PRIVATE_KERNEL_TEMPLATE_CHANGED")
        original_template = original_template.replace(needle, repl)
    compile(original_template, "<private_repaired_native_LTX>", "exec")
    dest = root / "private_resume_kernel"
    dest.mkdir(mode=0o700)
    (dest / "kernel_main.py").write_text(original_template)
    meta = {
        "id": KERNEL, "title": KERNEL.split("/")[1],
        "code_file": "kernel_main.py", "language": "python",
        "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    (dest / "kernel-metadata.json").write_text(json.dumps(meta))
    sealed = root / "STUDIO_WAVE_ORIGINAL_NEURAL_SEGMENT0.twenc"
    seal(clip, Path("ops/studio_wave_native_recipient_public.pem"), sealed,
         {"purpose": "studio-native-wave-checkpoint-0", "schema": 1,
          "source_sha256": SOURCE_SHA256, "segment_sha256": SEGMENT_SHA})
    (root / "RESUME_CPU_PREFLIGHT_SAFE.json").write_text(json.dumps({
        "original_kernel": KERNEL, "terminal_previous_state": "ERROR",
        "real_native_segment_0_sha256": SEGMENT_SHA,
        "frame_count": 33, "fps": 24, "resolution": [576, 1024],
        "original_segment_model_duration_seconds": measured,
        "original_segment_motion": motion,
        "original_segment_chroma": color["status"],
        "reused_existing_original_neural_frames": True,
        "additional_gpu_jobs_submitted_by_preflight": 0,
        "source_revision": revision,
        "private_repaired_kernel_prepared": True,
        "owner_encrypted_checkpoint_saved": sealed.is_file(),
        "human_artistic_approved": False, "published": False,
    }, sort_keys=True, indent=2))
    return {"ready": True, "real_frames_recovered": 33,
            "checkpoint_sha256": SEGMENT_SHA, "model_elapsed_seconds": measured,
            "private_kernel_prepared": True, "gpu_submissions": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--working", type=Path, required=True)
    ap.add_argument("--revision", required=True)
    args = ap.parse_args()
    report = prepare(args.working, revision=args.revision)
    print("STUDIO_WAVE_GENUINE_NATIVE_RESUME_SAFE_PREFLIGHT",
          json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
