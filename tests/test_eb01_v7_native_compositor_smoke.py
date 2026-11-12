"""CPU-only studio.hybrid_master integration smoke with explicitly FAKE media.

NOT production, NEVER a downloadable episode, NEVER model provenance.
Makes seven moving FFmpeg test patterns and six dummy sine WAVs in an
ephemeral folder; verifies release pipeline compatibility before GPU.
"""
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import studio.eb01_v7_storyboard as story
import tools.kaggle_eb01_v7_full_entry as full
from studio.comfy_neural import _probe

@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "ffmpeg required for CPU-only fixture test")
class StudioV7SevenSceneAssemblySmoke(unittest.TestCase):
 def test_six_voice_native_real_compositor_with_fixture_video(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   (root/"clips").mkdir()
   (root/"frames").mkdir()
   (root/"voices").mkdir()
   video_sha={}
   voice_sha={}
   for idx,(sid,who,frames,seconds,zoom,cx,cy,prompt) in enumerate(story.SHOTS):
    reference=root/"frames"/(sid+".png")
    from PIL import Image
    Image.new("RGB",(320,576),color=(38+idx,31+idx,35+idx)).save(reference)
    clip=root/"clips"/(sid+".mp4")
    subprocess.run(["ffmpeg","-hide_banner","-nostdin","-y","-v","error",
                    "-f","lavfi","-i",f"testsrc2=size=320x576:rate=24",
                    "-frames:v",str(frames),"-an","-c:v","libx264",
                    "-preset","ultrafast","-crf","24","-pix_fmt","yuv420p",str(clip)],
                   check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
    probe=_probe(clip)
    self.assertEqual(probe["frames"],frames)
    clip_sha=hashlib.sha256(clip.read_bytes()).hexdigest()
    source_sha=hashlib.sha256(reference.read_bytes()).hexdigest()
    proof={"schema":1,"engine":"neural_comfy","model":"wan22-5b",
           "reference_sha256":source_sha,"clip_sha256":clip_sha,
           "workflow_sha256":"a"*64,"video":probe,
           "inference_performed":True,"published":False,
           "note":"FAKE MEDIA ONLY FOR EPHEMERAL TEST, NOT AN ACTUAL MODEL INFERENCE"}
    (root/"clips"/(sid+".receipt.json")).write_text(json.dumps(proof))
    video_sha[sid]={"clip_sha256":clip_sha}
    if sid in story.VOICE_SECONDS:
     secs=story.VOICE_SECONDS[sid]
     rate=24000
     t=np.arange(round(secs*rate))/rate
     samples=(0.028*np.sin(2*np.pi*(200+idx*34)*t)).astype(np.float32)
     path=root/"voices"/(sid+".wav")
     sf.write(path,samples,rate)
     voice_sha[sid]=hashlib.sha256(path.read_bytes()).hexdigest()
   report=root/"private_receipt.json"
   with patch.object(story,"VOICE_SHA",voice_sha),patch.object(full,"ROOT",root),\
        patch.object(full,"FINAL",root/"fixture_not_final.mp4"),\
        patch.object(full,"FILM_QA",report):
    outcome=full.assemble_native({"private":True},
                 {"source_sha256":story.SRC_SHA},video_sha)
   self.assertEqual(outcome["real_I2V_shots"],7)
   self.assertEqual(outcome["russian_voice_lines"],6)
   self.assertEqual(outcome["subtitles"],6)
   self.assertEqual(outcome["native_studio"],"studio.hybrid_master.produce")
   self.assertTrue((root/"fixture_not_final.mp4").is_file())
   self.assertEqual((root/"fixture_not_final.srt").read_text().count("-->"),6)
   self.assertFalse(outcome["owner_artistic_approved"])
   self.assertFalse(outcome["human_dialogue_lipsync_approved"])

if __name__=="__main__":
 unittest.main()
