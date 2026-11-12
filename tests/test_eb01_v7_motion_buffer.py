"""True model-frame terminal-buffer QA, no model weights/GPU/paid services."""
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from studio.comfy_neural import _probe
from studio.eb01_v7_motion_buffer import MotionBufferRejected,finalize,digest
from studio.eb01_motion_artifact_gate import inspect_clip


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "CPU tests need ffmpeg/ffprobe")
class NativeWanBufferTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory()
  self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name)
  self.reference=self.root/"ref.png"
  Image.effect_noise((320,576),30).convert("RGB").save(self.reference)
  self.raw=self.root/"wan_original.mp4"
  self.proof=self.raw.with_suffix(".receipt.json")
  self.dest=self.root/"native_clean.mp4"

 def create(self,glitches=(18,19,20)):
  frames=[]
  for i in range(21):
   img=np.full((576,320,3),[30+i//2,29+i//2,27+i//2],dtype=np.uint8)
   if i in glitches:
    img[515:,:,:]=[255,214,7]
   frames.append(img.tobytes())
  proc=subprocess.run([
    "ffmpeg","-hide_banner","-nostdin","-y","-v","error",
    "-f","rawvideo","-pix_fmt","rgb24","-s","320x576","-r","24",
    "-i","-","-an","-c:v","libx264","-crf","16",
    "-pix_fmt","yuv420p",str(self.raw)
  ],input=b"".join(frames),capture_output=True,check=True,timeout=80)
  receipt={"schema":1,"engine":"neural_comfy","model":"wan22-5b",
           "reference_sha256":digest(self.reference),
           "clip_sha256":digest(self.raw),
           "workflow_sha256":"a"*64,
           "video":_probe(self.raw),"seed":42,"prompt_id":"real-contract-only",
           "inference_performed":True,"published":False,"human_approved":False}
  self.proof.write_text(json.dumps(receipt))

 def test_last_three_corrupt_raw_frames_discarded_but_first_17_real_kept(self):
  self.create()
  result=finalize(self.raw,self.proof,self.reference,self.dest,17)
  self.assertEqual(result["actual_frames"],17)
  self.assertEqual(result["original_frames"],21)
  self.assertEqual(result["status"],"REAL_NATIVE_WAN_FRAMES_CLEANED_OF_TERMINAL_BUFFER")
  self.assertEqual(inspect_clip(self.dest)["status"],"NO_NEON_ARTIFACT_FOUND")
  derived=json.loads(self.dest.with_suffix(".receipt.json").read_text())
  self.assertEqual(derived["derivation"]["original_Wan_raw_clip_sha256"],digest(self.raw))
  self.assertEqual(derived["derivation"]["kept_true_model_frames"],17)
  self.assertFalse(derived["human_approved"])
  with self.assertRaisesRegex(MotionBufferRejected,"overwrite"):
   finalize(self.raw,self.proof,self.reference,self.dest,17)

 def test_internal_glitch_blocks_story_and_no_fake_clean_output(self):
  self.create(glitches=(9,18,19,20))
  with self.assertRaisesRegex(MotionBufferRejected,"INSIDE"):
   finalize(self.raw,self.proof,self.reference,self.dest,17)
  self.assertFalse(self.dest.exists())

 def test_invalid_original_comfy_sha_does_not_forge_provenance(self):
  self.create(glitches=())
  data=json.loads(self.proof.read_text())
  data["clip_sha256"]="f"*64
  self.proof.write_text(json.dumps(data))
  with self.assertRaisesRegex(MotionBufferRejected,"No complete original"):
   finalize(self.raw,self.proof,self.reference,self.dest,17)
  self.assertFalse(self.dest.exists())


if __name__=="__main__":
 unittest.main()
