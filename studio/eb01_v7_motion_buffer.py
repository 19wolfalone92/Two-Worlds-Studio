"""Provenance-preserving Wan2.2 tail-buffer mastering, CPU-only.

Observed model run #43159416691 produced 49 native frames but final 3 had
neon strips. Rather than regenerate 7 costly shots, request four extra native
frames for each clip and discard them *only after* full-frame inspection.
Any glitch inside the 7-shot content window is a HARD production failure.

This is not frame interpolation, fake motion, repetition or a renderer fixture.
The final receipt explicitly records ORIGINAL neural clip + SHA and clean
CPU-only terminal crop; no pretending the re-encoded clip is the raw inference.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from studio.eb01_motion_artifact_gate import inspect_clip
from studio.comfy_neural import _probe


class MotionBufferRejected(RuntimeError):
    pass


def digest(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
 return h.hexdigest()


def finalize(raw:Path, raw_receipt:Path, reference:Path,
             dest:Path, expected_frames:int)->dict:
 raw,raw_receipt,reference,dest=map(Path,(raw,raw_receipt,reference,dest))
 if (dest.exists() or dest.is_symlink()
         or dest.with_suffix(".receipt.json").exists()
         or dest.with_suffix(".receipt.json").is_symlink()):
  raise MotionBufferRejected("Never overwrite previous model inference or edited output")
 if expected_frames<9 or expected_frames>237 or (expected_frames-1)%4:
  raise MotionBufferRejected("Expected native 24 FPS scene with Wan 4+1 framing")
 original=_probe(raw)
 receipt=json.loads(raw_receipt.read_text(encoding="utf-8"))
 if (original.get("frames")!=expected_frames+4
         or original.get("fps")!="24"
         or receipt.get("schema")!=1
         or receipt.get("engine")!="neural_comfy"
         or receipt.get("model")!="wan22-5b"
         or receipt.get("video")!=original
         or receipt.get("inference_performed") is not True
         or receipt.get("clip_sha256")!=digest(raw)
         or receipt.get("reference_sha256")!=digest(reference)
         or receipt.get("published") is not False
         or not re.fullmatch("[a-f0-9]{64}",str(receipt.get("workflow_sha256")))):
  raise MotionBufferRejected("No complete original Wan I2V source/receipt with pinned 4-frame buffer")

 scan=inspect_clip(raw)
 if scan["decoded_frames"]!=expected_frames+4:
  raise MotionBufferRejected("Cannot decode every source frame")
 # The only tolerable artifact is wholly in the 4 deliberately discarded
 # terminal buffer frames. A glitch in the intended scene means stop GPU.
 unexpected=[n for n in scan.get("flagged_frames",[]) if n<expected_frames]
 if unexpected or scan["status"]=="REJECT_NONTERMINAL_NEON_ARTIFACT":
  raise MotionBufferRejected("Wan corruption INSIDE useful actor-performance frames: "+str(unexpected))
 if scan["status"] not in ("NO_NEON_ARTIFACT_FOUND",
                           "TRIM_ONLY_TERMINAL_ARTIFACTS",
                           "REJECT_INSUFFICIENT_UNCORRUPTED_PILOT"):
  raise MotionBufferRejected("Unrecognized video corruption: "+scan["status"])

 dest.parent.mkdir(parents=True,exist_ok=True)
 try:
  subprocess.run([
   "ffmpeg","-hide_banner","-nostdin","-y","-loglevel","error","-xerror",
   "-i",str(raw),"-vf",f"trim=end_frame={expected_frames},setpts=PTS-STARTPTS",
   "-an","-r","24","-c:v","libx264","-preset","slow","-crf","17",
   "-pix_fmt","yuv420p","-movflags","+faststart",str(dest)
  ],check=True,capture_output=True,timeout=360)
  clean=_probe(dest)
  quality=inspect_clip(dest)
  if (clean["frames"]!=expected_frames or clean["fps"]!="24"
          or quality["status"]!="NO_NEON_ARTIFACT_FOUND"):
   raise MotionBufferRejected("Final original neural video window still corrupted or truncated")
  updated={
    **receipt,
    "clip_sha256":digest(dest),
    "video":clean,
    "published":False,"human_approved":False,
    "derivation":{
      "type":"ONLY_TERMINAL_FOUR_FRAME_BUFFER_TRIM",
      "original_Wan_raw_clip_sha256":digest(raw),
      "original_Comfy_receipt_sha256":digest(raw_receipt),
      "original_frames":expected_frames+4,
      "kept_true_model_frames":expected_frames,
      "dropped_terminal_frames":list(range(expected_frames,expected_frames+4)),
      "original_neon_artifact_frames":scan.get("flagged_frames",[]),
      "no_synthetic_frames_or_interpolation":True,
      "video_reencoded_for_mobile":True,
      "human_artistic_approval":False
    }
  }
  out_receipt=dest.with_suffix(".receipt.json")
  with out_receipt.open("x",encoding="utf-8") as f:
   json.dump(updated,f,ensure_ascii=False,indent=2)
  return {
    "status":"REAL_NATIVE_WAN_FRAMES_CLEANED_OF_TERMINAL_BUFFER",
    "source_clip_sha256":digest(raw),
    "source_receipt_sha256":digest(raw_receipt),
    "result_clip_sha256":digest(dest),
    "reference_sha256":digest(reference),
    "original_frames":original["frames"],
    "actual_frames":clean["frames"],"fps":clean["fps"],
    "artifact_diagnostic":scan["status"],
    "published":False,"human_approved":False
  }
 except BaseException:
  dest.unlink(missing_ok=True)
  dest.with_suffix(".receipt.json").unlink(missing_ok=True)
  raise
