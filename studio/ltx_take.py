"""Source-verifiable real LTX 2B image-to-video take importer.

Allows TWO WORLDS STUDIO to combine private, actually generated LTX and Wan
MP4s without labeling an LTX shot as a Wan/Blender model or substituting stills.
"""
from __future__ import annotations
import json,re
from pathlib import Path

from studio.comfy_neural import _probe
from studio.eb01_motion_artifact_gate import inspect_clip
from studio.release_gate import sha256

class LTXTakeRejected(ValueError):
    pass

def import_take(clip:Path, receipt:Path, reference:Path, *, duration:float)->dict:
    clip,receipt,reference=map(Path,(clip,receipt,reference))
    if not (clip.is_file() and receipt.is_file() and reference.is_file()):
        raise LTXTakeRejected("Missing original native video, genuine LTX receipt or locked character frame")
    if receipt.stat().st_size>80_000:
        raise LTXTakeRejected("LTX I2V provenance receipt is oversized")
    proof=json.loads(receipt.read_text(encoding="utf-8"))
    meta=_probe(clip)
    if (proof.get("schema")!=7 or proof.get("success") is not True
          or proof.get("shot")!="S02"
          or proof.get("status")!="ONE_REAL_LTX2B_S02_TECHNICALLY_VERIFIED_OWNER_REVIEW_REQUIRED"
          or proof.get("engine")!="ltx_official_fp16_t4_compatibility"
          or proof.get("model")!="Lightricks/LTX-Video-2B-0.9.6-distilled"
          or proof.get("official_source_git_sha")!="50f0aef1297574a4b29acbdb874e3a363a6d5f2e"
          or not re.fullmatch(r"[0-9a-f]{64}",str(proof.get("compatibility_patch_sha256")))
          or proof.get("original_reference_sha256")!=sha256(reference)
          or proof.get("clip_sha256")!=sha256(clip)
          or proof.get("true_gpu_neural_inference") is not True
          or proof.get("human_artistic_approved") is not False
          or proof.get("published") is not False
          or proof.get("invideo_used") is not False
          or proof.get("paid_inference_used") is not False
          or proof.get("frame_count")!=125
          or proof.get("fps")!=24
          or meta.get("frames")!=125 or meta.get("fps")!="24"
          or meta.get("width")!=320 or meta.get("height")!=576
          or meta.get("seconds",0)+0.001<duration):
        raise LTXTakeRejected("Unable to prove one genuine source-matched 125-frame LTX 2B video")
    quality=inspect_clip(clip)
    if quality["decoded_frames"]!=125 or quality["status"]!="NO_NEON_ARTIFACT_FOUND":
        raise LTXTakeRejected("RGB contamination in owner-review LTX scene")
    return {"status":"ACTUAL_GPU_LTX_I2V_SOURCE_VERIFIED",
      "model":proof["model"],"engine":"neural_ltx",
      "clip_sha256":proof["clip_sha256"],
      "original_reference_sha256":proof["original_reference_sha256"],
      "official_repository_revision":proof["official_source_git_sha"],
      "source_engine_fp16_patch_sha256":proof["compatibility_patch_sha256"],
      "actual_neural_frames":125,"synthetic_frames":0,
      "artistically_approved":False,"published":False}
