"""Isolated, fail-closed Wan2.2 T4 research pipeline; no FreeVideo/H3 code or weights.

Run from the repository root: python experiments/freevideo_t4/adaptive_pipeline.py plan
The only real inference path delegates to the existing, verified studio.comfy_neural.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKS = ("identity", "anatomy", "motion", "no_morphing")


class Blocked(RuntimeError):
    """A safety, compatibility, or production-quality gate was not passed."""


@dataclass(frozen=True)
class Hardware:
    name: str
    compute_capability: str
    vram_total_gib: float
    vram_free_gib: float
    source: str
    device_index: int = 0


def detect_hardware(index: int = 0) -> Hardware:
    """Probe a *single* real CUDA device; never sum memory across Kaggle workers."""
    try:
        import torch
        if not torch.cuda.is_available() or index >= torch.cuda.device_count() or index < 0:
            raise Blocked("No accessible CUDA GPU at the requested device index")
        props = torch.cuda.get_device_properties(index)
        with torch.cuda.device(index):
            free, total = torch.cuda.mem_get_info()
        return Hardware(props.name, f"{props.major}.{props.minor}",
                        round(total / 2**30, 3), round(free / 2**30, 3),
                        "detected_cuda", index)
    except Blocked:
        raise
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        raise Blocked(f"CUDA hardware probe unavailable: {type(exc).__name__}") from exc


def simulate_t4() -> Hardware:
    return Hardware("Tesla T4 (SIMULATED)", "7.5", 16.0, 14.0, "simulated_not_executable")


def choose_plan(hardware: Hardware) -> dict:
    """Conservative *heuristic*, not a measured Wan memory guarantee."""
    try:
        major, minor = (int(x) for x in hardware.compute_capability.split("."))
    except (ValueError, TypeError) as exc:
        raise Blocked("Malformed CUDA compute capability") from exc
    if major < 7:
        raise Blocked("Wan FP16 prototype needs CUDA compute capability >=7.0")
    if not math.isfinite(hardware.vram_free_gib) or hardware.vram_free_gib <= 0:
        raise Blocked("Free GPU VRAM measurement is invalid")
    reserved = max(.5, hardware.vram_free_gib * .05)
    budget = hardware.vram_free_gib - reserved
    if budget < 5:
        raise Blocked("Less than 5 GiB usable VRAM: no unverified GPU fallback")
    if budget >= 12:
        preview, final = (432, 768, 25, 20), (576, 1024, 49, 36)
        profile = "t4_16gb_conservative"
    else:
        preview, final = (288, 512, 25, 16), (432, 768, 49, 32)
        profile = "low_vram_cautious"
    def config(v):
        return dict(zip(("width", "height", "frames", "steps"), v))
    return {
        "schema": 1, "status": "HEURISTIC_PLAN_NOT_BENCHMARKED", "backend": "wan22-5b",
        "hardware": asdict(hardware), "profile": profile,
        "usable_vram_gib": round(budget, 2), "reserved_vram_gib": round(reserved, 2),
        "precision": "fp16", "bf16_allowed": major >= 8, "fps": 24,
        "preview": config(preview), "final": config(final),
        "excluded": ["freevideo_h3_not_t4_compatible", "vdn_h3_weights_not_used",
                     "no_cross_session_vram_pool", "no_bf16_on_t4",
                     "no_automatic_oom_retry", "no_automatic_publication"],
        "warnings": ["Memory sizes are estimates; installed ComfyUI and model weights are required.",
                     "Preview is a separate I2V generation, not a guaranteed low-res version of the final take.",
                     "Frame identity, bone anatomy and temporal coherence need human review."],
    }


def _hash_file(path: Path) -> str:
    dig = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def _private_path(path: Path, *, must_exist: bool = True) -> Path:
    p = path.expanduser().resolve()
    if p.is_relative_to(REPO_ROOT):
        raise Blocked("Keep references, GPU outputs and receipts OUTSIDE the public studio repository")
    if must_exist and (not p.is_file() or p.is_symlink()):
        raise Blocked(f"Required private file absent: {p}")
    return p


def _check_host(origin: str) -> None:
    u = urlsplit(origin)
    if u.scheme != "http" or u.username or u.password or u.query or u.fragment or u.path not in ("", "/"):
        raise Blocked("Only explicit loopback ComfyUI HTTP origins are permitted")
    try:
        if not ipaddress.ip_address(u.hostname or "").is_loopback:
            raise Blocked("ComfyUI must listen on loopback; use an SSH tunnel")
    except ValueError:
        if u.hostname != "localhost":
            raise Blocked("ComfyUI must listen on loopback")
    try:
        port = u.port
    except ValueError as exc:
        raise Blocked("Invalid ComfyUI port") from exc
    if port is None or not 1 <= port <= 65535:
        raise Blocked("Explicit ComfyUI port required")


def validate_review(review_path: Path, preview_video: Path, reference: Path) -> dict:
    """Proof that a *human* reviewed the exact preview, not just a filename."""
    review = _private_path(review_path)
    preview = _private_path(preview_video)
    if preview.suffix.lower() != ".mp4":
        raise Blocked("Preview must be an MP4")
    if review.stat().st_size > 32_768:
        raise Blocked("Review JSON too large")
    try:
        data = json.loads(review.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise Blocked("Invalid human review JSON") from exc
    if not isinstance(data, dict) or data.get("schema") != 1 or data.get("approved") is not True:
        raise Blocked("Explicit, approved human review is required for final rendering")
    if not isinstance(data.get("reviewer"), str) or not data["reviewer"].strip():
        raise Blocked("Human review must identify its reviewer")
    if data.get("preview_sha256") != _hash_file(preview):
        raise Blocked("Preview SHA-256 differs from the reviewed video")
    if data.get("reference_sha256") != _hash_file(reference):
        raise Blocked("Reference SHA-256 differs from the reviewed source")
    if not isinstance(data.get("checks"), dict) or any(data["checks"].get(k) is not True for k in CHECKS):
        raise Blocked("Identity, anatomy, motion and deformation checks must all be approved")
    return data


def _history_append(path: Path, entry: dict) -> None:
    destination = _private_path(path, must_exist=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # O_APPEND makes each bounded record an atomic append on local filesystems.
    payload = (json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    if len(payload) > 4096:
        raise Blocked("Oversized benchmark entry")
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(fd, payload) != len(payload):
            raise OSError("Partial statistics write")
        os.fsync(fd)
    finally:
        os.close(fd)


def summary(history: Path) -> dict:
    path = _private_path(history)
    groups = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("result") != "GENERATED_REVIEW_REQUIRED":
            continue
        key = (r.get("hardware"), r.get("stage"), r.get("resolution"), r.get("frames"))
        groups.setdefault(key, []).append(float(r["elapsed_s"]))
    return {"status": "OBSERVED_BENCHMARKS_ONLY", "groups": [
        {"hardware": k[0], "stage": k[1], "resolution": k[2], "frames": k[3],
         "count": len(v), "median_elapsed_s": round(statistics.median(v), 2)}
        for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))],
        "warning": "Different resolutions, prompts and hardware are not directly comparable."}


def run(args: argparse.Namespace) -> dict:
    if args.command == "stats":
        return summary(Path(args.history))
    if args.command == "plan":
        hardware = simulate_t4() if args.simulate_t4 else detect_hardware(args.device)
        return choose_plan(hardware)
    # Real execution always re-probes *real* hardware, never a caller supplied spec.
    if not args.execute or not args.accept_model_license:
        raise Blocked("GPU generation requires BOTH --execute and --accept-model-license")
    hardware = detect_hardware(args.device)
    plan = choose_plan(hardware)
    reference = _private_path(Path(args.reference))
    if reference.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} or reference.stat().st_size > 32 * 2**20:
        raise Blocked("Reference must be a local image <=32 MiB")
    output = _private_path(Path(args.output), must_exist=False)
    if output.suffix.lower() != ".mp4" or output.exists() or output.is_symlink():
        raise Blocked("Output must be a new private .mp4 path")
    if output.with_suffix(".receipt.json").exists():
        raise Blocked("Refusing to replace an existing proof receipt")
    if not isinstance(args.prompt, str) or not 12 <= len(args.prompt.strip()) <= 1800:
        raise Blocked("Motion direction must contain 12–1800 characters")
    _check_host(args.host)
    if args.command == "final":
        if not args.review or not args.preview_video:
            raise Blocked("Final requires --review and --preview-video")
        validate_review(Path(args.review), Path(args.preview_video), reference)
    stage = plan[args.command]
    command = [sys.executable, "-m", "studio.comfy_neural", "--model", "wan22-5b",
               "--reference", str(reference), "--output", str(output),
               "--prompt", args.prompt, "--seed", str(args.seed),
               "--width", str(stage["width"]), "--height", str(stage["height"]),
               "--frames", str(stage["frames"]), "--steps", str(stage["steps"]),
               "--host", args.host, "--timeout", str(args.timeout),
               "--accept-model-license", "--execute-local-gpu"]
    start = time.monotonic()
    try:
        p = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True,
                           timeout=args.timeout + 120, check=False)
    except subprocess.TimeoutExpired as exc:
        raise Blocked("Studio inference timed out; check ComfyUI for a still-running job before retrying") from exc
    if p.returncode != 0:
        raise Blocked("Existing studio GPU bridge refused the operation: " + p.stdout[-1500:] + p.stderr[-500:])
    try:
        result = json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        raise Blocked("Studio bridge did not return a valid JSON result") from exc
    if result.get("status") != "GENERATED_REVIEW_REQUIRED" or result.get("human_approved") is not False:
        raise Blocked("Studio bridge did not return an unapproved, verified neural take")
    if not output.is_file() or result.get("clip_sha256") != _hash_file(output):
        raise Blocked("Generated clip hash mismatch")
    proof = output.with_suffix(".receipt.json")
    if not proof.is_file() or json.loads(proof.read_text(encoding="utf-8")).get("clip_sha256") != result["clip_sha256"]:
        raise Blocked("GPU proof receipt missing or mismatched")
    elapsed = round(time.monotonic() - start, 3)
    if args.history:
        _history_append(Path(args.history), {
            "result": result["status"], "stage": args.command,
            "backend": "wan22-5b", "hardware": hardware.name,
            "resolution": f'{stage["width"]}x{stage["height"]}',
            "frames": stage["frames"], "elapsed_s": elapsed,
            "clip_sha256": result["clip_sha256"], "reference_sha256": _hash_file(reference),
            "published": False, "human_approved": False})
    return {"status": "GENERATED_REVIEW_REQUIRED", "stage": args.command,
            "output": str(output), "receipt": str(proof), "elapsed_s": elapsed,
            "clip_sha256": result["clip_sha256"], "published": False, "human_approved": False}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Two Worlds isolated FreeVideo-inspired T4 research (Wan2.2 only)")
    p.add_argument("command", choices=("plan", "preview", "final", "stats"))
    p.add_argument("--simulate-t4", action="store_true", help="offline PLAN ONLY; never permits execution")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--reference")
    p.add_argument("--output")
    p.add_argument("--prompt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--host", default="http://127.0.0.1:8188")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--review", help="human-approved private JSON for final")
    p.add_argument("--preview-video", help="exact preview approved in --review")
    p.add_argument("--history", help="private JSONL measured run stats")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--accept-model-license", action="store_true")
    args = p.parse_args(argv)
    try:
        if args.simulate_t4 and args.command != "plan":
            raise Blocked("Simulation is only supported for plan, never generation")
        if args.command in {"preview", "final"} and not all((args.reference, args.output, args.prompt)):
            raise Blocked("--reference, --output and --prompt are required")
        if args.command == "stats" and not args.history:
            raise Blocked("--history is required for stats")
        if not 60 <= args.timeout <= 14400 or not 0 <= args.seed < 2**32:
            raise Blocked("Invalid timeout or seed")
        result = run(args)
        code = 0
    except (Blocked, OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"status": "BLOCKED", "reason": str(exc), "published": False}
        code = 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
