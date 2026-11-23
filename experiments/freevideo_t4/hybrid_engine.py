"""Experimental Freevideo hybrid entrypoint: private photo -> Studio Wan bridge -> QC.

Uses existing studio.video_ai_plan, studio.comfy_neural, studio.video_qc
through the isolated adaptive_pipeline. No cloud inference, paid model API,
auto-publication, auto-approval or production studio writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from PIL import Image, ImageOps, ImageStat

try:  # Works both as a script and as a package imported by tests.
    from . import adaptive_pipeline
except ImportError:
    import adaptive_pipeline

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1
PROMPT_MAX = 1800
ID = re.compile(r"[a-z0-9][a-z0-9_-]{2,59}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
DEFAULT_SIZE = (864, 1536)


class HybridBlocked(ValueError):
    """Fail closed rather than spending GPU, changing source, or faking a video."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def private_path(raw: str | Path, *, existing: bool = True) -> Path:
    original = Path(raw).expanduser().absolute()
    if original.is_symlink() or original.is_relative_to(ROOT):
        raise HybridBlocked("Inputs and outputs must remain private, outside public Git checkout")
    path = original.resolve()
    if path.is_relative_to(ROOT):
        raise HybridBlocked("Private input resolved into public repository")
    if existing:
        if not path.is_file() or path.is_symlink():
            raise HybridBlocked("Required private regular file is missing")
    elif path.exists() or path.is_symlink():
        raise HybridBlocked("Refusing to overwrite an existing private file")
    return path


def _json(path: Path) -> dict:
    path = private_path(path)
    if path.stat().st_size > 256 * 1024:
        raise HybridBlocked("JSON manifest/receipt exceeds 256 KiB")
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise HybridBlocked("Duplicate JSON keys are not allowed")
            d[k] = v
        return d
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HybridBlocked("Cannot parse private JSON") from exc
    if not isinstance(data, dict):
        raise HybridBlocked("JSON must contain an object")
    return data


def _save_new_json(path: Path, data: dict) -> None:
    private_path(path, existing=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def prepare(source: str | Path, workdir: str | Path, job_id: str, prompt: str,
            *, seed: int = 42, size: tuple[int, int] = DEFAULT_SIZE) -> dict:
    """Stage a lossless-content, *uncropped* private 9:16 reference.

    Letterbox instead of pretending to reconstruct cropped hands/feet/face.
    This is image preprocessing, NOT generative video.
    """
    source = private_path(source)
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HybridBlocked("Private source must be PNG/JPEG/WebP")
    if source.stat().st_size > 32 * 1024 * 1024:
        raise HybridBlocked("Reference exceeds 32 MiB")
    if not isinstance(job_id, str) or not ID.fullmatch(job_id):
        raise HybridBlocked("Job id must be 3-60 lowercase safe characters")
    if not isinstance(prompt, str) or not 12 <= len(prompt.strip()) <= PROMPT_MAX:
        raise HybridBlocked("Motion prompt must have 12-1800 characters")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise HybridBlocked("Invalid random seed")
    if size != DEFAULT_SIZE:
        raise HybridBlocked("Experimental hybrid uses fixed 864x1536 9:16 reference")
    root = Path(workdir).expanduser().absolute()
    if root.is_relative_to(ROOT) or root.is_symlink() or root.exists():
        raise HybridBlocked("Choose a NEW private job directory outside public repository")
    if root.resolve().is_relative_to(ROOT):
        raise HybridBlocked("Private job directory resolves into public repository")

    # Decode completely before making a job directory; no side effects on a bad input.
    try:
        with Image.open(source) as orig:
            orig = ImageOps.exif_transpose(orig)
            orig.load()
            if min(orig.size) < 300 or max(orig.size) > 8192:
                raise HybridBlocked("Reference dimensions outside 300-8192px bounds")
            if orig.mode in ("RGBA", "LA") and orig.getchannel("A").getextrema()[0] != 255:
                raise HybridBlocked("Transparent source has no reviewed scene background")
            rgb = orig.convert("RGB")
            # Preserve every original pixel region at the same scale; do NOT face-crop.
            scale = min(size[0] / rgb.width, size[1] / rgb.height)
            target = (max(1, round(rgb.width * scale)),
                      max(1, round(rgb.height * scale)))
            thumb = rgb.resize(target, Image.Resampling.LANCZOS)
            # Neutral padding avoids hallucinated/duplicated objects around subject.
            samples = [rgb.crop((0, 0, min(24, rgb.width), min(24, rgb.height))),
                       rgb.crop((max(0, rgb.width-24), 0, rgb.width, min(24, rgb.height)))]
            means = [ImageStat.Stat(tile).mean for tile in samples]
            neutral = tuple(int(max(0, min(255, sum(v[c] for v in means) / len(means) * .40)))
                            for c in range(3))
            canvas = Image.new("RGB", size, neutral)
            position = ((size[0]-target[0]) // 2, (size[1]-target[1]) // 2)
            canvas.paste(thumb, position)
    except (OSError, ValueError) as exc:
        if isinstance(exc, HybridBlocked):
            raise
        raise HybridBlocked("Reference image could not be decoded") from exc

    from studio.video_ai_plan import _reference
    root.mkdir(mode=0o700, parents=True)
    try:
        reference = root / "reference_9x16.png"
        with reference.open("xb") as handle:
            canvas.save(handle, format="PNG", optimize=True)
        _reference(reference)  # Production Studio's actual native reference contract.
        manifest = {
            "schema": SCHEMA, "engine": "freevideo_hybrid", "job_id": job_id,
            "backend": "wan22-5b", "neural_inference": "NOT_RUN",
            "source_path": str(source), "source_sha256": sha256(source),
            "reference_path": str(reference), "reference_sha256": sha256(reference),
            "prompt": prompt.strip(), "seed": seed,
            "source_size": list(rgb.size), "reference_size": list(size),
            "resized_content_size": list(target), "content_origin": list(position),
            "framing": "contain_without_cropping", "human_approved": False,
            "published": False, "privacy": "private_only"
        }
        _save_new_json(root / "job.json", manifest)
    except BaseException:
        # Never delete the original user-supplied photo; only our own unfinished dir.
        if (root / "job.json").exists():
            (root / "job.json").unlink()
        (root / "reference_9x16.png").unlink(missing_ok=True)
        try:
            root.rmdir()
        except OSError:
            pass
        raise
    return {"status": "PREPARED_NO_GPU", "job_file": str(root / "job.json"),
            "source_sha256": manifest["source_sha256"],
            "reference_sha256": manifest["reference_sha256"],
            "framing": manifest["framing"], "model": "wan22-5b",
            "human_approved": False, "published": False}


def _load_job(job_file: str | Path) -> tuple[Path, dict]:
    path = private_path(job_file)
    doc = _json(path)
    if (doc.get("schema") != SCHEMA or doc.get("engine") != "freevideo_hybrid"
            or doc.get("backend") != "wan22-5b" or doc.get("published") is not False
            or doc.get("human_approved") is not False or doc.get("privacy") != "private_only"
            or not isinstance(doc.get("job_id"), str) or not ID.fullmatch(doc["job_id"])):
        raise HybridBlocked("Invalid/mixed project manifest")
    if type(doc.get("seed")) is not int or not 0 <= doc["seed"] < 2**32:
        raise HybridBlocked("Invalid persisted seed")
    if not isinstance(doc.get("prompt"), str) or not 12 <= len(doc["prompt"]) <= PROMPT_MAX:
        raise HybridBlocked("Invalid persisted prompt")
    src = private_path(doc["source_path"])
    ref = private_path(doc["reference_path"])
    if ref != path.parent / "reference_9x16.png":
        raise HybridBlocked("Job reference escapes its immutable private job directory")
    if any(not isinstance(doc.get(k), str) or not HEX.fullmatch(doc[k])
           for k in ("source_sha256", "reference_sha256")):
        raise HybridBlocked("Invalid persisted SHA-256")
    if sha256(src) != doc["source_sha256"] or sha256(ref) != doc["reference_sha256"]:
        raise HybridBlocked("Source or staged reference changed after preparation")
    from studio.video_ai_plan import _reference
    _reference(ref)
    return path, doc


def run_stage(job_file: str | Path, stage: str, *,
              execute: bool = False, accept_model_license: bool = False,
              weights_manifest: str | None = None, device: int = 0,
              host: str = "http://127.0.0.1:8188", timeout: int = 3600,
              review: str | None = None) -> dict:
    path, doc = _load_job(job_file)
    if stage not in ("preview", "final"):
        raise HybridBlocked("Only preview and final neural stages are supported")
    if not execute or not accept_model_license:
        raise HybridBlocked("Neural GPU inference requires explicit execution and licence consent")
    if not weights_manifest:
        raise HybridBlocked("Private --weights-manifest is required")
    if type(device) is not int or not 0 <= device <= 31:
        raise HybridBlocked("Invalid GPU device index")
    out = path.parent / (stage + ".mp4")
    if any(p.exists() or p.is_symlink() for p in
           (out, out.with_suffix(".receipt.json"), out.with_suffix(".freevideo.json"),
            out.with_suffix(".hybrid.json"))):
        raise HybridBlocked("Stage has existing output or receipt; never re-render blindly")
    if stage == "final" and review is None:
        raise HybridBlocked("Human-reviewed preview receipt required before final")
    # Reuse existing fail-closed Freevideo adaptive planner / native Studio
    # ComfyUI video bridge. No alternate cloud generator can be selected.
    args = argparse.Namespace(
        command=stage, execute=True, accept_model_license=True, device=device,
        reference=doc["reference_path"], output=str(out), prompt=doc["prompt"],
        seed=doc["seed"], timeout=timeout, host=host,
        history=str(path.parent / "timings.jsonl"),
        weights_manifest=weights_manifest, simulate_t4=False,
        review=review, preview_video=str(path.parent / "preview.mp4") if stage == "final" else None)
    result = adaptive_pipeline.run(args)
    if result.get("status") != "GENERATED_REVIEW_REQUIRED":
        raise HybridBlocked("Neural Studio bridge did not confirm a generated MP4")
    # Native Studio QC and Freevideo QC must remain conceptually separate:
    # numerical motion differences cannot certify identity/real anatomy.
    secondary = None
    try:
        from studio.video_qc import sample_video
        secondary = sample_video(out)
    except Exception as exc:
        secondary = {"status": "REVIEW_REQUIRED", "error_type": type(exc).__name__,
                     "artistic_approval": False}
    output_doc = {
        "schema": 1, "engine": "freevideo_hybrid", "job_id": doc["job_id"],
        "stage": stage, "backend": "wan22-5b", "model_path": "studio.comfy_neural",
        "reference_sha256": doc["reference_sha256"],
        "clip_sha256": result["clip_sha256"],
        "source_sha256": doc["source_sha256"],
        "freevideo_receipt_sha256": sha256(out.with_suffix(".freevideo.json")),
        "studio_receipt_sha256": sha256(out.with_suffix(".receipt.json")),
        "studio_video_qc": secondary,
        "neural_inference": "GPU_BRIDGE_REPORTED",
        "visual_quality": "NOT_VERIFIED",
        "human_approved": False, "published": False,
    }
    _save_new_json(out.with_suffix(".hybrid.json"), output_doc)
    return {"status": "GENERATED_REVIEW_REQUIRED", "stage": stage,
            "clip": str(out), "hybrid_receipt": str(out.with_suffix(".hybrid.json")),
            "clip_sha256": result["clip_sha256"],
            "visual_quality": "NOT_VERIFIED", "published": False}


def review_template(job_file: str | Path) -> dict:
    path, doc = _load_job(job_file)
    preview = path.parent / "preview.mp4"
    if not preview.is_file():
        raise HybridBlocked("Actual generated preview.mp4 is required for visual review")
    bridge = _json(preview.with_suffix(".receipt.json"))
    hybrid = _json(preview.with_suffix(".hybrid.json"))
    if (hybrid.get("clip_sha256") != sha256(preview)
            or bridge.get("clip_sha256") != sha256(preview)
            or hybrid.get("source_sha256") != doc["source_sha256"]
            or hybrid.get("reference_sha256") != doc["reference_sha256"]):
        raise HybridBlocked("Preview provenance invalid, do not create approval template")
    target = path.parent / "human_review_TEMPLATE.json"
    draft = {"schema": 1, "approved": False, "reviewer": "",
             "preview_sha256": sha256(preview),
             "reference_sha256": doc["reference_sha256"],
             "checks": dict.fromkeys(adaptive_pipeline.CHECKS, False)}
    _save_new_json(target, draft)
    return {"status": "VISUAL_REVIEW_REQUIRED", "template": str(target),
            "warning": "Never set checks true without watching the actual generated clip."}


def status(job_file: str | Path) -> dict:
    path, doc = _load_job(job_file)
    takes = {}
    for stage in ("preview", "final"):
        clip = path.parent / (stage + ".mp4")
        hybrid_path = clip.with_suffix(".hybrid.json")
        if hybrid_path.is_file():
            receipt = _json(hybrid_path)
            if (not clip.is_file() or receipt.get("clip_sha256") != sha256(clip)
                    or receipt.get("stage") != stage or receipt.get("job_id") != doc["job_id"]
                    or receipt.get("visual_quality") != "NOT_VERIFIED"
                    or receipt.get("human_approved") is not False):
                raise HybridBlocked("Stage output is tampered or has false quality status")
            takes[stage] = {"status": "GENERATED_REVIEW_REQUIRED",
                            "sha256": receipt["clip_sha256"],
                            "visual_quality": "NOT_VERIFIED"}
        elif clip.exists():
            takes[stage] = {"status": "INCOMPLETE_RECEIPT_RECONCILE_REQUIRED"}
        else:
            takes[stage] = {"status": "NOT_GENERATED"}
    return {"status": "PRIVATE_HYBRID_JOB", "job_id": doc["job_id"],
            "source_sha256": doc["source_sha256"], "takes": takes,
            "gpu": "UNVERIFIED_BY_STATUS_COMMAND", "published": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Private Freevideo + Studio hybrid I2V; Kaggle T4 only")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("prepare")
    init.add_argument("--source", required=True)
    init.add_argument("--workdir", required=True)
    init.add_argument("--job-id", required=True)
    init.add_argument("--prompt", required=True)
    init.add_argument("--seed", type=int, default=42)
    for action in ("preview", "final", "review-template", "status"):
        cmd = sub.add_parser(action)
        cmd.add_argument("--job", required=True)
        if action in ("preview", "final"):
            cmd.add_argument("--execute", action="store_true")
            cmd.add_argument("--accept-model-license", action="store_true")
            cmd.add_argument("--weights-manifest")
            cmd.add_argument("--device", type=int, default=0)
            cmd.add_argument("--host", default="http://127.0.0.1:8188")
            cmd.add_argument("--timeout", type=int, default=3600)
            if action == "final":
                cmd.add_argument("--review")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.source, args.workdir, args.job_id, args.prompt, seed=args.seed)
        elif args.command in ("preview", "final"):
            result = run_stage(args.job, args.command, execute=args.execute,
                               accept_model_license=args.accept_model_license,
                               weights_manifest=args.weights_manifest,
                               device=args.device, host=args.host, timeout=args.timeout,
                               review=getattr(args, "review", None))
        elif args.command == "review-template":
            result = review_template(args.job)
        else:
            result = status(args.job)
        code = 0
    except (HybridBlocked, adaptive_pipeline.Blocked, OSError, ValueError,
            RuntimeError, KeyError, TypeError) as exc:
        result, code = {"status": "BLOCKED", "error_type": type(exc).__name__,
                        "reason": str(exc), "gpu_success": False, "published": False}, 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
