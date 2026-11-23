"""Freevideo (T4): isolated, guarded Wan2.2 I2V research pipeline.

Engineering ideas inspired by FreeVideo; NO FreeVideo/VDN/H3 model, code or weights.
Nothing is imported into the production Studio. GPU inference is explicit only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import subprocess
import sys
import time
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKS = ("identity", "anatomy", "motion", "no_morphing")
MODEL = "wan22-5b"
MAX_STATS_BYTES = 32 * 1024 * 1024


class Blocked(RuntimeError):
    """A hardware, provenance, security, or visual review gate failed."""


@dataclass(frozen=True)
class Hardware:
    name: str
    compute_capability: str
    vram_total_gib: float
    vram_free_gib: float
    source: str
    device_index: int = 0


def detect_hardware(index: int = 0) -> Hardware:
    """Read one *actual* CUDA device; no pooled memory or simulated execution."""
    try:
        import torch
        if type(index) is not int or index < 0 or not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise Blocked("No accessible CUDA GPU at requested index")
        p = torch.cuda.get_device_properties(index)
        with torch.cuda.device(index):
            free, total = torch.cuda.mem_get_info()
        return Hardware(p.name, f"{p.major}.{p.minor}", round(total / 2**30, 3),
                        round(free / 2**30, 3), "detected_cuda", index)
    except Blocked:
        raise
    except (ImportError, RuntimeError, OSError, AttributeError) as exc:
        raise Blocked(f"CUDA hardware probe unavailable: {type(exc).__name__}") from exc


def simulate_t4() -> Hardware:
    return Hardware("Tesla T4 (SIMULATED)", "7.5", 16.0, 14.0, "simulated_not_executable")


def _is_t4(hardware: Hardware) -> bool:
    return bool(re.search(r"\bT4\b", hardware.name, re.I)) and hardware.compute_capability == "7.5"


def choose_plan(hardware: Hardware) -> dict:
    """Workload heuristic, NOT a benchmark or a promise that a model fits VRAM."""
    try:
        major, minor = (int(x) for x in hardware.compute_capability.split("."))
    except (ValueError, TypeError) as exc:
        raise Blocked("Malformed CUDA compute capability") from exc
    if major < 7:
        raise Blocked("Prototype needs CUDA capability >= 7.0")
    if (not math.isfinite(hardware.vram_total_gib) or not math.isfinite(hardware.vram_free_gib)
            or hardware.vram_total_gib <= 0 or hardware.vram_free_gib <= 0
            or hardware.vram_free_gib > hardware.vram_total_gib + .1):
        raise Blocked("Invalid reported GPU VRAM")
    reserved = max(.5, hardware.vram_free_gib * .05)
    budget = hardware.vram_free_gib - reserved
    if budget < 5:
        raise Blocked("Less than 5 GiB usable VRAM; fail closed")
    if budget >= 12:
        preview, final = (432, 768, 25, 20), (576, 1024, 49, 36)
        profile = "t4_16gb_conservative"
    else:
        preview, final = (288, 512, 25, 16), (432, 768, 49, 32)
        profile = "low_vram_cautious"
    def config(v):
        return dict(zip(("width", "height", "frames", "steps"), v))
    return {
        "schema": 2, "status": "HEURISTIC_PLAN_NOT_BENCHMARKED", "backend": MODEL,
        "hardware": asdict(hardware), "profile": profile,
        "usable_vram_gib": round(budget, 2), "reserved_vram_gib": round(reserved, 2),
        "precision": "fp16", "bf16_allowed": major >= 8, "fps": 24,
        "preview": config(preview), "final": config(final),
        "excluded": ["freevideo_h3_not_t4_compatible", "vdn_h3_weights_not_used",
                     "no_cross_session_vram_pool", "no_bf16_on_t4",
                     "no_automatic_oom_retry", "no_automatic_publication"],
        "warnings": ["Estimated VRAM only: real Wan weights/ComfyUI may still OOM.",
                     "Preview is a separately generated 1-second take, not an upscaled final.",
                     "Final is a separate 2-second take and requires its own visual review.",
                     "Execution only allows T4 and a matching, local ComfyUI device."]}


@contextmanager
def device_lease(index: int):
    """Reject simultaneous Freevideo jobs on one T4 in the *same* Kaggle worker.

    Not distributed across independent Kaggle sessions; no GPU work is done here.
    """
    if type(index) is not int or not 0 <= index <= 31:
        raise Blocked("Invalid CUDA device index for per-worker lock")
    try:
        import fcntl
    except ImportError as exc:
        raise Blocked("GPU locking requires Linux fcntl (Kaggle worker)") from exc
    folder = Path(tempfile.gettempdir()) / f"freevideo-t4-lock-{os.getuid()}"
    try:
        folder.mkdir(mode=0o700, exist_ok=True)
        st = folder.lstat()
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise Blocked("GPU lock directory has unsafe ownership or permissions")
        handle = os.open(folder / f"gpu-{index}.lock",
                         os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except (OSError, PermissionError) as exc:
        raise Blocked(f"Could not create private GPU lock: {type(exc).__name__}") from exc
    try:
        info = os.fstat(handle)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise Blocked("GPU lock file ownership/permissions are unsafe")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Blocked("This GPU is busy with another Freevideo job; no concurrent render") from exc
        yield
    finally:
        os.close(handle)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_path(path: Path, *, must_exist: bool = True) -> Path:
    """Reject both lexical public paths and symlink escapes into or out of repo."""
    original = path.expanduser().absolute()
    if original.is_relative_to(REPO_ROOT) or original.is_symlink():
        raise Blocked("Private assets may not be in Git checkout or symlinked")
    resolved = original.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        raise Blocked("Private assets must not resolve into public Git checkout")
    if must_exist and (not resolved.is_file() or resolved.is_symlink()):
        raise Blocked(f"Missing private file: {resolved}")
    return resolved


def _json_file(path: Path, *, limit: int = 32768) -> dict:
    if path.stat().st_size > limit:
        raise Blocked("Proof JSON exceeds allowed size")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Blocked("Proof JSON is invalid") from exc
    if not isinstance(obj, dict):
        raise Blocked("Proof JSON must be an object")
    return obj


def _write_proof(path: Path, obj: dict) -> None:
    private = _private_path(path, must_exist=False)
    private.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        private.unlink(missing_ok=True)
        raise


def _check_host(origin: str) -> None:
    u = urlsplit(origin)
    if u.scheme != "http" or u.username or u.password or u.query or u.fragment or u.path not in ("", "/"):
        raise Blocked("ComfyUI must use an explicit loopback-only HTTP origin")
    try:
        is_loopback = ipaddress.ip_address(u.hostname or "").is_loopback
    except ValueError:
        is_loopback = u.hostname == "localhost"
    if not is_loopback:
        raise Blocked("ComfyUI must be local loopback (or an explicitly configured tunnel)")
    try:
        port = u.port
    except ValueError as exc:
        raise Blocked("Invalid ComfyUI port") from exc
    if port is None or not 1 <= port <= 65535:
        raise Blocked("An explicit ComfyUI port is required")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise Blocked("ComfyUI redirection is forbidden")


def attest_backend(origin: str, hardware: Hardware) -> dict:
    """Compare selected worker GPU with local ComfyUI's reported CUDA device.

    Never assume local torch memory and an SSH-tunneled remote device are the same.
    This is a consistency guard, not a cryptographic attestation.
    """
    _check_host(origin)
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(Request(origin.rstrip("/") + "/system_stats"), timeout=7) as resp:
            if int(resp.headers.get("Content-Length", "0")) > 131072:
                raise Blocked("ComfyUI system stats too large")
            raw = resp.read(131073)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise Blocked(f"ComfyUI hardware check unavailable: {type(exc).__name__}") from exc
    if len(raw) > 131072:
        raise Blocked("ComfyUI hardware response too large")
    try:
        info = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise Blocked("Invalid ComfyUI system stats") from exc
    devices = info.get("devices") if isinstance(info, dict) else None
    if not isinstance(devices, list):
        raise Blocked("ComfyUI did not report CUDA devices")
    matches = [d for d in devices if isinstance(d, dict) and d.get("type") == "cuda"
               and d.get("index") == hardware.device_index and
               re.search(r"\bT4\b", str(d.get("name", "")), flags=re.I)]
    if len(matches) != 1 or not _is_t4(hardware):
        raise Blocked("ComfyUI GPU differs from local selected Tesla T4")
    remote_total = matches[0].get("vram_total")
    if (not isinstance(remote_total, (int, float)) or not math.isfinite(remote_total)
            or abs(remote_total / 2**30 - hardware.vram_total_gib) > 1.5):
        raise Blocked("ComfyUI VRAM differs from selected local GPU")
    return {"device_index": hardware.device_index, "gpu": str(matches[0]["name"]),
            "status": "CONSISTENT_NOT_CRYPTOGRAPHIC_ATTESTATION"}


def _video_probe(path: Path) -> dict:
    """Decode the entire stream frame clock, rather than trust metadata alone."""
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-count_frames", "-show_entries",
                            "stream=width,height,avg_frame_rate,nb_read_frames",
                            "-of", "json", str(path)], capture_output=True, text=True,
                           timeout=60, check=True)
        videos = json.loads(p.stdout).get("streams", [])
        if len(videos) != 1:
            raise ValueError("Expected one decoded video stream")
        video = videos[0]
        fps = Fraction(video["avg_frame_rate"])
        if fps <= 0:
            raise ValueError("Invalid FPS")
        result = {"width": int(video["width"]), "height": int(video["height"]),
                  "fps": str(fps), "frames": int(video["nb_read_frames"])}
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, ZeroDivisionError, TypeError) as exc:
        raise Blocked("Cannot verify decoded preview video frame clock") from exc
    return result


def validate_review(review_path: Path, preview_video: Path, reference: Path,
                    *, prompt: str | None = None, seed: int | None = None) -> dict:
    """Bind human review to source, real clip, bridge receipt and our stage receipt.

    A JSON 'approved' flag is still a manual assertion, NOT proof of vision.
    """
    review = _json_file(_private_path(review_path))
    preview = _private_path(preview_video)
    if preview.suffix.lower() != ".mp4":
        raise Blocked("Preview must be an MP4")
    source_hash, clip_hash = _hash_file(reference), _hash_file(preview)
    if review.get("schema") != 1 or review.get("approved") is not True:
        raise Blocked("Human approval is required to run final stage")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise Blocked("Human reviewer name is required")
    if review.get("preview_sha256") != clip_hash or review.get("reference_sha256") != source_hash:
        raise Blocked("Preview/reference SHA-256 differs from reviewed source")
    if not isinstance(review.get("checks"), dict) or any(review["checks"].get(k) is not True for k in CHECKS):
        raise Blocked("Identity, anatomy, movement, and no-morphing checks must be manually approved")
    bridge_path = _private_path(preview.with_suffix(".receipt.json"))
    bridge = _json_file(bridge_path)
    if (bridge.get("schema") != 1 or bridge.get("engine") != "neural_comfy"
            or bridge.get("model") != MODEL or bridge.get("clip_sha256") != clip_hash
            or bridge.get("reference_sha256") != source_hash
            or bridge.get("inference_performed") is not True
            or bridge.get("human_approved") is not False
            or bridge.get("published") is not False
            or not re.fullmatch(r"[0-9a-f]{64}", str(bridge.get("workflow_sha256", "")))):
        raise Blocked("Preview does not have authentic Studio bridge provenance")
    stage = _json_file(_private_path(preview.with_suffix(".freevideo.json")))
    if (stage.get("schema") != 2 or stage.get("stage") != "preview"
            or stage.get("clip_sha256") != clip_hash
            or stage.get("reference_sha256") != source_hash
            or stage.get("bridge_receipt_sha256") != _hash_file(bridge_path)
            or stage.get("backend") != MODEL or stage.get("human_approved") is not False):
        raise Blocked("Preview lacks matching Freevideo stage provenance")
    if prompt is not None and stage.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest():
        raise Blocked("Final prompt differs from approved preview")
    if seed is not None and stage.get("seed") != seed:
        raise Blocked("Final seed differs from approved preview")
    if type(bridge.get("seed")) is not int or bridge["seed"] != stage.get("seed"):
        raise Blocked("Studio seed differs from recorded Freevideo preview")
    measured = _video_probe(preview)
    params = stage.get("geometry")
    if (not isinstance(params, dict) or
            any(measured.get(k) != params.get(k) for k in ("width", "height", "frames"))
            or measured["fps"] != "24"):
        raise Blocked("Preview decoded video differs from recorded stage geometry")
    from_bridge = bridge.get("video")
    if (not isinstance(from_bridge, dict) or
            any(measured.get(k) != from_bridge.get(k) for k in ("width", "height", "frames"))
            or from_bridge.get("fps") != "24"):
        raise Blocked("Studio neural receipt differs from actual decoded preview")
    return review


def _history_append(path: Path, entry: dict) -> None:
    destination = _private_path(path, must_exist=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(payload) > 4096:
        raise Blocked("Oversized statistics record")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(destination, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise Blocked("Metrics destination must be a regular private file")
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError("Partial metrics write")
        os.fsync(fd)
    finally:
        os.close(fd)


def summary(history: Path) -> dict:
    path = _private_path(history)
    if path.stat().st_size > MAX_STATS_BYTES:
        raise Blocked("Metrics file exceeds safe read limit")
    groups: dict[tuple, list[float]] = {}
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if len(line) > 4096:
                raise Blocked(f"Oversized metrics record at line {number}")
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise Blocked(f"Invalid metrics JSON at line {number}") from exc
            if not isinstance(r, dict) or r.get("result") != "GENERATED_REVIEW_REQUIRED":
                raise Blocked(f"Unrecognized metrics record at line {number}")
            elapsed = r.get("elapsed_s")
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
                raise Blocked(f"Invalid execution time at line {number}")
            if r.get("backend") != MODEL or r.get("stage") not in ("preview", "final"):
                raise Blocked(f"Unknown backend or stage in metrics at line {number}")
            key = (r.get("hardware"), r.get("stage"), r.get("resolution"), r.get("frames"))
            if any(not isinstance(k, (str, int)) for k in key):
                raise Blocked(f"Invalid grouping field at line {number}")
            groups.setdefault(key, []).append(elapsed)
            count += 1
    return {"status": "OBSERVED_BENCHMARKS_ONLY", "successful_runs": count,
            "groups": [{"hardware": k[0], "stage": k[1], "resolution": k[2],
                        "frames": k[3], "count": len(v),
                        "median_elapsed_s": round(statistics.median(v), 2)}
                       for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))],
            "warning": "Times are observed, not an apples-to-apples speed comparison."}


def _run_unlocked(args: argparse.Namespace) -> dict:
    if args.command == "stats":
        return summary(Path(args.history))
    if args.command == "plan":
        h = simulate_t4() if args.simulate_t4 else detect_hardware(args.device)
        return choose_plan(h)
    # Authorization must precede hardware probing, model loading or any GPU call.
    if not args.execute or not args.accept_model_license:
        raise Blocked("Generation requires BOTH --execute and --accept-model-license")
    hardware = detect_hardware(args.device)
    if hardware.source != "detected_cuda" or not _is_t4(hardware):
        raise Blocked("Freevideo experimental runtime permits only a real NVIDIA Tesla T4")
    plan = choose_plan(hardware)
    reference = _private_path(Path(args.reference))
    # Lock the source before invoking a potentially long-running neural backend.
    # A modified source while rendering would otherwise break review provenance.
    reference_hash_before_gpu = _hash_file(reference)
    if reference.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} or reference.stat().st_size > 32 * 2**20:
        raise Blocked("Reference must be a private image <=32 MiB")
    output = _private_path(Path(args.output), must_exist=False)
    if output.suffix.lower() != ".mp4" or output.exists() or output.is_symlink():
        raise Blocked("Output must be a new private MP4 filename")
    for suffix in (".receipt.json", ".freevideo.json"):
        if output.with_suffix(suffix).exists() or output.with_suffix(suffix).is_symlink():
            raise Blocked("Refusing to overwrite an existing proof")
    if not isinstance(args.prompt, str) or not 12 <= len(args.prompt.strip()) <= 1800:
        raise Blocked("Motion prompt must be 12–1800 characters")
    _check_host(args.host)
    if args.command == "final":
        if not args.review or not args.preview_video:
            raise Blocked("Final requires a private human --review and --preview-video")
        validate_review(Path(args.review), Path(args.preview_video), reference,
                        prompt=args.prompt, seed=args.seed)
    stage = plan[args.command]
    # The local PyTorch probe does NOT prove ComfyUI is using the same device.
    # Deny when backend cannot corroborate GPU identity/index and total VRAM.
    attestation = attest_backend(args.host, hardware)
    command = [sys.executable, "-m", "studio.comfy_neural", "--model", MODEL,
               "--reference", str(reference), "--output", str(output),
               "--prompt", args.prompt, "--seed", str(args.seed),
               "--width", str(stage["width"]), "--height", str(stage["height"]),
               "--frames", str(stage["frames"]), "--steps", str(stage["steps"]),
               "--host", args.host, "--timeout", str(args.timeout),
               "--accept-model-license", "--execute-local-gpu"]
    start = time.monotonic()
    try:
        process = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True,
                                 timeout=args.timeout + 120, check=False)
    except subprocess.TimeoutExpired as exc:
        raise Blocked(f"GPU timeout; inspect the existing ComfyUI job before retrying. Output: {output}") from exc
    if process.returncode != 0:
        raise Blocked(f"Studio bridge refused this run (possibly still running); inspect before retry: "
                      + process.stdout[-1000:] + process.stderr[-500:])
    try:
        result = json.loads(process.stdout)
    except (ValueError, UnicodeError) as exc:
        raise Blocked(f"Studio returned invalid JSON; inspect generated files before retry: {output}") from exc
    if not isinstance(result, dict) or result.get("status") != "GENERATED_REVIEW_REQUIRED" or result.get("human_approved") is not False:
        raise Blocked(f"Studio bridge did not confirm a verified, unapproved take: {output}")
    if not output.is_file() or result.get("clip_sha256") != _hash_file(output):
        raise Blocked(f"Generated video hash mismatch; do not retry blindly: {output}")
    if _hash_file(reference) != reference_hash_before_gpu:
        raise Blocked(f"Reference image changed during GPU inference; quarantine this take: {output}")
    bridge_path = output.with_suffix(".receipt.json")
    bridge = _json_file(_private_path(bridge_path))
    if (bridge.get("schema") != 1 or bridge.get("engine") != "neural_comfy"
            or not re.fullmatch(r"[0-9a-f]{64}", str(bridge.get("workflow_sha256", "")))
            or bridge.get("clip_sha256") != result["clip_sha256"]
            or bridge.get("reference_sha256") != reference_hash_before_gpu
            or bridge.get("model") != MODEL or bridge.get("inference_performed") is not True
            or type(bridge.get("seed")) is not int or bridge["seed"] != args.seed
            or bridge.get("human_approved") is not False or bridge.get("published") is not False):
        raise Blocked(f"Generated video receipt mismatch; inspect before retry: {output}")
    measured = _video_probe(output)
    from_bridge = bridge.get("video")
    if (measured["frames"] != stage["frames"] or measured["width"] != stage["width"]
            or measured["height"] != stage["height"] or measured["fps"] != "24"
            or not isinstance(from_bridge, dict)
            or any(from_bridge.get(k) != measured[k] for k in ("frames", "width", "height", "fps"))):
        raise Blocked(f"Generated video differs from planned geometry or Studio receipt: {output}")
    elapsed = round(max(time.monotonic() - start, .001), 3)
    stage_proof = {
        "schema": 2, "stage": args.command, "backend": MODEL, "human_approved": False,
        "published": False, "weights_verified": bridge.get("weights_verified") is True,
        "reference_sha256": reference_hash_before_gpu,
        "clip_sha256": result["clip_sha256"],
        "bridge_receipt_sha256": _hash_file(bridge_path),
        "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
        "seed": args.seed, "geometry": dict(stage), "fps": 24,
        "hardware": asdict(hardware), "attestation": attestation,
        "elapsed_s": elapsed, "evidence": "REAL_BRIDGE_RECEIPT_NO_QUALITY_CLAIM"}
    # Do not re-render if a secondary proof write fails: MP4 already exists.
    _write_proof(output.with_suffix(".freevideo.json"), stage_proof)
    history_warning = None
    if args.history:
        try:
            _history_append(Path(args.history), {
                "result": "GENERATED_REVIEW_REQUIRED", "stage": args.command,
                "backend": MODEL, "hardware": hardware.name,
                "resolution": f'{stage["width"]}x{stage["height"]}', "frames": stage["frames"],
                "elapsed_s": elapsed, "clip_sha256": result["clip_sha256"],
                "reference_sha256": _hash_file(reference), "published": False,
                "human_approved": False})
        except (OSError, Blocked, ValueError) as exc:
            # Metrics are optional; never report an existing valid MP4 as a failed generation.
            history_warning = f"Generated clip is valid but telemetry could not be saved: {type(exc).__name__}"
    return {"status": "GENERATED_REVIEW_REQUIRED", "stage": args.command,
            "output": str(output), "receipt": str(bridge_path),
            "stage_receipt": str(output.with_suffix(".freevideo.json")),
            "elapsed_s": elapsed, "clip_sha256": result["clip_sha256"],
            "history_warning": history_warning, "human_approved": False, "published": False}


def run(args: argparse.Namespace) -> dict:
    """Permit only one explicitly authorized GPU job per device and worker."""
    if args.command in ("preview", "final"):
        if not args.execute or not args.accept_model_license:
            raise Blocked("Generation requires BOTH --execute and --accept-model-license")
        with device_lease(args.device):
            return _run_unlocked(args)
    return _run_unlocked(args)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Freevideo isolated T4-only experiment (Wan2.2, NOT H3)")
    parser.add_argument("command", choices=("plan", "preview", "final", "stats"))
    parser.add_argument("--simulate-t4", action="store_true", help="PLAN ONLY, never executes GPU")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--reference")
    parser.add_argument("--output")
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--review", help="external human-approved review JSON")
    parser.add_argument("--preview-video", help="exact approved preview.mp4")
    parser.add_argument("--history", help="private append-only actual timings")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-model-license", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.simulate_t4 and args.command != "plan":
            raise Blocked("Simulation is only permitted for plan, never for execution")
        if args.command in ("preview", "final") and not all((args.reference, args.output, args.prompt)):
            raise Blocked("--reference, --output and --prompt are required")
        if args.command == "stats" and not args.history:
            raise Blocked("--history is required for stats")
        if args.device < 0 or not 60 <= args.timeout <= 14400 or not 0 <= args.seed < 2**32:
            raise Blocked("Invalid device, timeout or seed")
        result = run(args)
        code = 0
    except (Blocked, OSError, ValueError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
        result = {"status": "BLOCKED", "reason": str(exc), "published": False}
        code = 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
