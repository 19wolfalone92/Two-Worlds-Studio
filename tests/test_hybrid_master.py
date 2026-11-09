"""Contract + real FFmpeg smoke tests for the optional hybrid editor."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from studio.hybrid_master import check_manifest, produce, subtitles


class HybridMasterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # FFmpeg makes a real media fixture without external assets or models.
        self.image = self.root / "shot.png"
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (360, 640), "#243f5b")
        draw = ImageDraw.Draw(image)
        draw.ellipse((40, 220, 280, 460), fill="#f2ba7f")
        image.save(self.image)
        self.data = {"schema": 1, "width": 360, "height": 640, "fps": 12,
                     "shots": [{"image": "shot.png", "duration": 1.0,
                                "camera": "push_in", "caption": "Привет"}]}
        self.path = self.root / "manifest.json"

    def write(self):
        self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")

    def test_preflight_and_subtitles(self):
        p = check_manifest(self.data, self.root)
        self.assertEqual(p["frames"], 12)
        self.assertIn("00:00:01,000", subtitles(p["shots"]))
        self.assertEqual(p["duration"], 1)

    def test_wrong_aspect_and_duration_rejected(self):
        self.data["width"] = 640
        with self.assertRaisesRegex(ValueError, "9:16"):
            check_manifest(self.data, self.root)
        self.data["width"] = 360
        self.data["shots"][0]["duration"] = 999
        with self.assertRaisesRegex(ValueError, "duration"):
            check_manifest(self.data, self.root)

    def test_traversal_and_unapproved_master_rejected(self):
        self.data["shots"][0]["image"] = "../outside.png"
        with self.assertRaisesRegex(ValueError, "out-of-project"):
            check_manifest(self.data, self.root)
        self.data["shots"][0]["image"] = "shot.png"
        with self.assertRaisesRegex(ValueError, "approvals"):
            check_manifest(self.data, self.root, master=True)
        self.data.update(editorial_approved=True, rights_approved=True)
        with self.assertRaisesRegex(ValueError, "not explicitly approved"):
            check_manifest(self.data, self.root, master=True)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not installed")
    def test_real_video_two_modes_and_audio(self):
        clip = self.root / "clip.mp4"
        wav = self.root / "voice.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                        "-i", "color=c=red:s=360x640:r=12:d=0.6", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(clip)], check=True)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                        "-i", "sine=frequency=330:duration=0.6", str(wav)], check=True)
        self.data["shots"][0]["voice"] = "voice.wav"
        self.data["shots"].append({"clip": "clip.mp4", "duration": 1.0,
                                   "camera": "static", "caption": "Вторая сцена"})
        self.write()
        info = produce(self.path, self.root / "final.mp4")
        self.assertEqual(info["motion_clips"], 1)
        self.assertEqual(info["static_camera_shots"], 1)
        self.assertEqual(info["audio_tracks"], 1)
        self.assertEqual(info["review_status"], "preview_unreviewed")
        self.assertEqual(info["duration_seconds"], 2)
        self.assertTrue((self.root / "final.mp4").stat().st_size > 10_000)
        self.assertIn("00:00:02,000", (self.root / "final.srt").read_text())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not installed")
    def test_silent_output_has_audio_stream(self):
        self.write()
        receipt = produce(self.path, self.root / "silent.mp4")
        self.assertEqual(receipt["duration_seconds"], 1)
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                 "stream=codec_type", "-of", "json", str(self.root / "silent.mp4")],
                                capture_output=True, check=True, text=True)
        kinds = {x["codec_type"] for x in json.loads(result.stdout)["streams"]}
        self.assertEqual(kinds, {"audio", "video"})


if __name__ == "__main__":
    unittest.main()
