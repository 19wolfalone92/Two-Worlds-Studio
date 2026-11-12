"""Crash-safe PRIVATE checkpoint of one genuine Wan I2V source before artistic QC.

Observed incident: private S02 129-frame Wan inference took 2806 seconds,
then MotionBufferRejected flagged 12 content frames and the caller returned
ERROR WITHOUT persisting its raw MP4 (the original lived in /kaggle/temp).
The correct recovery is to preserve the genuine raw clip/receipt in
/kaggle/working BEFORE running artistic QA, and quarantine uncertain frames
for direct visual review rather than silently accept or destroy them.

This module NEVER represents a quarantined video as a finished scene or film.
It does NOT generate video or spend GPU tokens.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from studio.comfy_neural import _probe


class CheckpointRejected(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fd:
        for block in iter(lambda: fd.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_raw(raw: Path, original_receipt: Path, reference: Path,
                   private_output: Path, sid: str, expected_frames: int) -> dict:
    """Verify bytes + immutable model receipt and persist raw clip privately.

    Caller MUST call immediately after genuine model inference and BEFORE
    any potentially rejecting image/artistic QA, including terminal-buffer QA.
    """
    if sid not in {"S02", "S03", "S04", "S05", "S06"}:
        raise CheckpointRejected("Existing completed S00/S01 must not be regenerated")
    src, receipt, ref, dest_dir = map(
        Path, (raw, original_receipt, reference, private_output))
    if not src.is_file() or not receipt.is_file() or not ref.is_file():
        raise CheckpointRejected("No genuine model MP4, original receipt or pinned reference")
    if expected_frames < 9 or (expected_frames - 1) % 4:
        raise CheckpointRejected("Bad expected Wan I2V frames")
    video = _probe(src)
    proof = json.loads(receipt.read_text(encoding="utf-8"))
    if (video.get("frames") != expected_frames + 4
            or video.get("fps") != "24"
            or proof.get("schema") != 1
            or proof.get("engine") != "neural_comfy"
            or proof.get("model") != "wan22-5b"
            or proof.get("video") != video
            or proof.get("inference_performed") is not True
            or proof.get("clip_sha256") != _sha(src)
            or proof.get("reference_sha256") != _sha(ref)
            or proof.get("published") is not False):
        raise CheckpointRejected("Original model source, image SHA or decode proof mismatch")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{sid}.original_wan_raw.mp4"
    dest_proof = dest_dir / f"{sid}.original_wan_raw.receipt.json"
    checkpoint = dest_dir / f"{sid}.original_wan_raw.checkpoint.json"
    if any(p.exists() or p.is_symlink() for p in (dest, dest_proof, checkpoint)):
        raise CheckpointRejected("Never overwrite any existing original native media")
    staged = []
    try:
        for source, target in ((src, dest), (receipt, dest_proof)):
            with tempfile.NamedTemporaryFile(
                    dir=dest_dir, prefix=f".{sid}.", suffix=".part",
                    delete=False) as fd:
                temp = Path(fd.name)
                staged.append(temp)
                with source.open("rb") as inp:
                    shutil.copyfileobj(inp, fd, length=2 * 1024 * 1024)
                fd.flush()
                os.fsync(fd.fileno())
            if _sha(temp) != _sha(source):
                raise CheckpointRejected("Original MP4 or model receipt changed during checkpoint")
            os.replace(temp, target)
            staged.remove(temp)
        out = {
            "schema": 7,
            "status": "GENUINE_NEURAL_SOURCE_PRESERVED_QA_PENDING",
            "shot": sid, "model": "wan22-5b",
            "original_mp4": dest.name,
            "original_mp4_sha256": _sha(dest),
            "original_model_receipt": dest_proof.name,
            "original_model_receipt_sha256": _sha(dest_proof),
            "original_reference_sha256": _sha(ref),
            "raw_frames": video["frames"],
            "usable_frame_target": expected_frames,
            "model_inference_executed": True,
            "human_artistic_approval": False,
            "can_use_in_finished_episode": False,
            "private_only": True, "published": False
        }
        with checkpoint.open("x", encoding="utf-8") as fd:
            json.dump(out, fd, ensure_ascii=False, indent=2)
        return out
    except BaseException:
        for p in staged:
            p.unlink(missing_ok=True)
        # Do not erase valid completed outputs on post-copy receipt failure.
        raise


def mark_quarantined(private_output: Path, sid: str, reason: str,
                     inspection: dict | None = None) -> dict:
    """Write a separate non-approval marker, keep every original raw byte."""
    root = Path(private_output)
    receipt = root / f"{sid}.original_wan_raw.checkpoint.json"
    if not receipt.is_file():
        raise CheckpointRejected("No authentic raw checkpoint: cannot quarantine")
    state = json.loads(receipt.read_text(encoding="utf-8"))
    path = root / state["original_mp4"]
    if not path.is_file() or _sha(path) != state["original_mp4_sha256"]:
        raise CheckpointRejected("Raw Wan movie missing or altered while recording QC failure")
    report = {
        "schema": 7,
        "status": "RAW_MODEL_CLIP_QUARANTINED_FOR_ARTISTIC_REVIEW",
        "shot": sid, "reason": reason[:450],
        "source_sha256": state["original_mp4_sha256"],
        "source_frame_count": state["raw_frames"],
        "inspection": inspection or {},
        "needs_manual_visual_review": True,
        "automatically_approved": False,
        "full_episode_ready": False,
        "published": False,
    }
    target = root / f"{sid}.needs_visual_review.json"
    with target.open("x", encoding="utf-8") as fd:
        json.dump(report, fd, ensure_ascii=False, indent=2)
    return report


def selected_contact_sheet(raw: Path, review: Path, frame_indices: list[int],
                           columns: int = 4) -> dict:
    """Create only a SMALL private sample of flagged and nearby frames."""
    from PIL import Image, ImageDraw
    if review.exists() or review.is_symlink():
        raise CheckpointRejected("Never overwrite private QC review")
    if len(frame_indices) < 1 or len(frame_indices) > 20 or min(frame_indices) < 0:
        raise CheckpointRejected("Need 1..20 nonnegative original frame indices")
    nframes = int(_probe(Path(raw))["frames"])
    selected = sorted(set(frame_indices))
    if max(selected) >= nframes:
        raise CheckpointRejected("Selected frame is outside real source")
    cell_w, cell_h, footer = 160, 288, 26
    rows = (len(selected) + columns - 1) // columns
    canvas = Image.new("RGB", (cell_w * columns, (cell_h + footer) * rows), (24,21,23))
    draw = ImageDraw.Draw(canvas)
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder)
        for i, frame_id in enumerate(selected):
            frame = base / f"{frame_id:04d}.png"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-xerror",
                "-i", str(raw), "-vf", f"select=eq(n\\,{frame_id}),scale={cell_w}:{cell_h}",
                "-vsync", "vfr", "-frames:v", "1", str(frame)
            ], check=True, capture_output=True, timeout=60)
            with Image.open(frame) as im:
                canvas.paste(im.convert("RGB"), ((i % columns) * cell_w,
                                                (i // columns) * (cell_h + footer)))
            draw.text(((i % columns) * cell_w + 5,
                       (i // columns) * (cell_h + footer) + cell_h + 3),
                      f"RAW FRAME {frame_id}", fill=(240,227,213))
    canvas.save(review, format="JPEG", quality=85, optimize=True)
    return {"status": "LOW_RES_PRIVATE_WAN_FRAME_INSPECTION_ONLY",
            "frames": selected, "sha256": _sha(review),
            "model_art_approved": False, "published": False}
