import json
import shutil
import tempfile
import unittest
from pathlib import Path

from studio.render import validate_episode, gradient, animate_frame, make_audio, render


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.assets = Path(self.tmp.name)
        self.scene = {"id":"S01","duration":1.0,"palette":"earth","effect":"mirror",
                      "heading":"WHO ARE YOU?","caption":"An impossible reflection."}

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid(self):
        self.assertEqual(len(validate_episode({"scenes":[self.scene]},self.assets)),1)

    def test_duplicate_ids_forbidden(self):
        with self.assertRaises(ValueError):
            validate_episode({"scenes":[self.scene,self.scene]},self.assets)

    def test_path_traversal_forbidden(self):
        self.scene["background"] = "../outside.png"
        with self.assertRaises(ValueError):
            validate_episode({"scenes":[self.scene]},self.assets)

    def test_bad_effect_forbidden(self):
        self.scene["effect"] = "infinite magic"
        with self.assertRaises(ValueError):
            validate_episode({"scenes":[self.scene]},self.assets)

    def test_frame_dimensions(self):
        size=(180,320)
        base=gradient(size,"earth")
        frame=animate_frame(base,self.scene,.3,3,12)
        self.assertEqual(frame.size,size)
        self.assertEqual(frame.mode,"RGB")

    @unittest.skipUnless(shutil.which("ffmpeg"),"ffmpeg unavailable")
    def test_short_mp4(self):
        episode = self.assets/"sample.json"
        episode.write_text(json.dumps({"scenes":[self.scene]}),encoding="utf-8")
        video = self.assets/"clip.mp4"
        result=render(episode,video,self.assets,180,320,6)
        self.assertTrue(video.exists())
        self.assertGreater(video.stat().st_size,1000)
        self.assertEqual(result["frames"],6)


if __name__ == "__main__":
    unittest.main()
