"""Generate SHORT genuine Wan2.2 native actor takes to avoid long-take latent collapse.

Source is the existing original private Elizabeth/Kostya image & six SHA-locked
Russian Qwen voices. We deliberately DO NOT fake one 89/125-frame inference.
Each segment is its own genuine ComfyUI I2V shot with a model receipt.
A full scene may be assembled only from QC-passed native frames, with every
segment source and cut recorded in the resulting audit.

Kaggle launched through exact SHA-pinned GitHub CI. No publish. No repeat
of canonical S00/S01/S03/S04/S05.
"""
from __future__ import annotations
import base64
import json
import sys

CONFIG_B64="__EB01_V7_MICRO_SHOT_CONFIG_B64__"

# Wan2.2 I2V requires count 4n+1. Four extra guard frames are generated.
# S06 original variant1 has visible neon starting at frame 65; the first
# 60 pixels frames are independently recoverable once provenance is checked.
# 60 clean model frames + 29 new native frames = exactly 89 S06 frames.
# S02: 41 + 41 + 45 = 127 model frames, trim ONLY final 2 good frames.
SEGMENT={
 "S06_TAIL29":{
   "shot":"S06","frames":29,"seed_variant":2,
   "prompt":("Cinematic tight speaking close-up of the SAME anatomically exact "
      "ivory-bone skeleton gentleman in an unchanged matte black suit and "
      "black bow tie in the warmly lit Victorian interview office. "
      "He turns his skull slightly, bony lower jaw speaks and settles into "
      "a very subtle humorous grin. Natural coherent 3D actor movement, "
      "no new props, no changing bones, no colored neon, no glowing bands, "
      "no glitches, no skin, NO RGB streaks.")
 },
 "S02_A41":{
   "shot":"S02","frames":41,"seed_variant":1,
   "prompt":("First part of a job interview answer. The SAME bone skeleton "
      "gentleman in an identical fitted black three piece suit and bow tie "
      "answers the interviewer clearly and confidently with natural moving "
      "jaw, neck and shoulders. Slight skull tilt. Warm brown walnut room, "
      "steady chiaroscuro lighting, no neon, no color flashes.")
 },
 "S02_B41":{
   "shot":"S02","frames":41,"seed_variant":2,
   "prompt":("Second camera angle of the SAME skeleton gentleman interview: "
      "he raises one skeletal hand gently while speaking, the jaw moves "
      "naturally and the black suit stays correctly fitted, cinematic warm "
      "Victorian office background, subtle natural camera composition. "
      "No RGB streaks or strange multi-colored room elements.")
 },
 "S02_C45":{
   "shot":"S02","frames":45,"seed_variant":1,
   "prompt":("Final close-up of the SAME elegant anatomically accurate "
      "skeleton Kostya, the name answer finishes with an understated "
      "comic nod. Authentic ivory skull, natural bony jaw articulation "
      "and subtle neck movement, unchanged black bow tie, warm amber "
      "lighting, consistent dark gothic office, no flesh or colored bands.")
 }
}

def main():
    cfg=json.loads(base64.b64decode(CONFIG_B64,validate=True))
    segment=cfg.get("segment")
    if segment not in SEGMENT:
        raise ValueError("Unapproved exact native micro-scene")
    spec=SEGMENT[segment]
    if (cfg.get("shot")!=spec["shot"] or
         cfg.get("quality_retake") is not True or
         cfg.get("seed_variant")!=spec["seed_variant"] or
         cfg.get("allow_social_publish") is not False):
        raise ValueError("Segment differs from exact approved micro-shot")
    import studio.eb01_v7_storyboard as story
    rows=[]
    for row in story.SHOTS:
        if row[0]==spec["shot"]:
            row=tuple([row[0],row[1],spec["frames"],row[3],
                       row[4],row[5],row[6],spec["prompt"]])
        rows.append(row)
    story.SHOTS=tuple(rows)
    from tools import kaggle_eb01_v7_single_shot as worker
    worker.CONFIG_B64=CONFIG_B64
    print("EB01_GENUINE_MICRO_SHOT",segment,"frames",spec["frames"],
          "source",spec["shot"],"seed_variant",spec["seed_variant"],flush=True)
    worker.main()
    # Normal worker independently checkpoints the original and checks every
    # decoded frame. Mark micro-segment in a supplemental immutable receipt.
    from pathlib import Path
    work=Path("/kaggle/working")
    receipt=work/"EB01_V7_SINGLE_SHOT_JOB.json"
    result=json.loads(receipt.read_text(encoding="utf-8"))
    if result.get("success") is not True or result.get("frame_count")!=spec["frames"]:
        raise RuntimeError("Native micro-shot has not passed source/QC checks")
    (work/"EB01_V7_MICRO_SHOT_RECEIPT.json").write_text(
        json.dumps({"schema":7,"segment":segment,"shot":spec["shot"],
        "original_scene_frames":{"S02":125,"S06":89}[spec["shot"]],
        "verified_native_segment_frames":spec["frames"],
        "quality_pass":True,"source_clip_sha256":result["clip_sha256"],
        "no_interpolated_frames":True,"published":False,
        "artistically_approved":False},indent=2)+"\n")
    print("EB01_GENUINE_MICRO_COMPLETE",segment,result["clip_sha256"],flush=True)

if __name__=="__main__":
    main()
