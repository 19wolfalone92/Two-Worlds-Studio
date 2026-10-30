"""Cinematic promo gate: CPU-only regression suite."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from studio.promo_quality_gate import check_manifest, inspect_video, InvalidPromo


class PromoGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        Image.new("RGB",(720,1280),(44,18,36)).save(self.path/"still.png")
        self.doc={"schema":1,"id":"promo","language":"ru",
                  "platform":"instagram_reels","music_review":"human_listened",
                  "final_card_text":"ELIZABETH BONES","call_to_action":"Direct",
                  "shots":[{"world":w,"kind":"still","asset":"still.png","seconds":5,
                            "rights_review":"human_verified","contains_real_deceased":False}
                           for w in ("comic","care","comic")]}
    def write(self):
        m=self.path/"manifest.json"
        m.write_text(json.dumps(self.doc),encoding="utf8")
        return m
    def test_slideshow_never_ready(self):
        r=check_manifest(self.write())
        self.assertEqual(r["status"],"BLOCKED")
        self.assertFalse(r["ready_to_publish"])
        self.assertTrue(any("motion" in s for s in r["blocks"]))
    def test_path_traversal_rejected(self):
        self.doc["shots"][0]["asset"]="../private.png"
        with self.assertRaises(InvalidPromo):
            check_manifest(self.write())
    def test_deceased_exclusion_required(self):
        del self.doc["shots"][0]["contains_real_deceased"]
        r=check_manifest(self.write())
        self.assertTrue(any("real deceased" in s for s in r["blocks"]))
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"ffmpeg required")
    def test_music_only_still_fails_visual_review(self):
        video=self.path/"still.mp4"
        subprocess.run(["ffmpeg","-nostdin","-v","error","-y",
          "-loop","1","-framerate","12","-i",str(self.path/"still.png"),
          "-f","lavfi","-i","anullsrc=r=44100:cl=stereo",
          "-t","2.5","-c:v","libx264","-preset","ultrafast",
          "-pix_fmt","yuv420p","-c:a","aac",str(video)],check=True)
        r=inspect_video(video)
        self.assertEqual(r["status"],"WARNINGS")
        self.assertGreater(r["stationary_ratio"],.85)


if __name__=="__main__":
    unittest.main()
