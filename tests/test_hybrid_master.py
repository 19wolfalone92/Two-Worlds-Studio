"""Contract + real FFmpeg smoke tests for the optional hybrid editor."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from studio.hybrid_master import check_manifest, produce, subtitles, ass_subtitles


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

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_mobile_profile_encodes_the_selected_native_frame_clock(self):
        for key in ("width", "height", "fps"):
            self.data.pop(key)
        self.data["delivery_profile"] = "mobile"
        plan = check_manifest(self.data, self.root)
        self.assertEqual((plan["width"], plan["height"], plan["fps"], plan["frames"]), (576, 1024, 24, 24))
        self.write()
        result = produce(self.path, self.root / "mobile.mp4")
        self.assertEqual(result["delivery_profile"], "mobile")
        self.assertEqual(result["resolution"], [576, 1024])
        self.assertEqual(result["fps"], 24)
        self.data["width"] = 1080
        with self.assertRaisesRegex(ValueError, "Mobile delivery"):
            check_manifest(self.data, self.root)

    def test_real_phrase_timestamps_win_over_full_shot_caption(self):
        self.data["shots"][0]["caption"] = "Весь кадр"
        self.data["shots"][0]["subtitle_cues"] = [
            {"start": 0.1, "end": 0.3, "text": "Мама"},
            {"start": 0.4, "end": 0.8, "text": "Смотри сюда"},
        ]
        plan = check_manifest(self.data, self.root)
        text = subtitles(plan["shots"])
        self.assertIn("00:00:00,100 --> 00:00:00,300", text)
        self.assertIn("00:00:00,400 --> 00:00:00,800", text)
        self.assertNotIn("Весь кадр", text)
        self.assertIn("Смотри сюда", text)
        self.data["shots"][0]["subtitle_cues"][1]["start"] = 0.2
        with self.assertRaisesRegex(ValueError, "overlapping"):
            check_manifest(self.data, self.root)

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

    def test_ass_captions_escape_override_tags(self):
        self.data["shots"][0]["caption"] = "{\\pos(4,5)}Тайна"
        plan = check_manifest(self.data, self.root)
        ass = ass_subtitles(plan["shots"], plan["width"], plan["height"])
        self.assertIn("PlayResX: 360", ass)
        self.assertIn("Тайна", ass)
        self.assertNotIn("{\\pos", ass)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not installed")
    def test_real_burned_cyrillic_subtitles_and_music_ducking(self):
        # Actual FFmpeg render; inputs are synthetic tests, never production art.
        wav = self.root / "voice.wav"
        music = self.root / "music.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", "sine=frequency=330:duration=0.7",
                        "-ar", "48000", str(wav)], check=True)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", "sine=frequency=220:duration=1.5",
                        "-ar", "48000", str(music)], check=True)
        self.data.update(burn_subtitles=True, music="music.wav", music_volume=.28)
        self.data["shots"][0]["voice"] = "voice.wav"
        self.data["shots"][0]["caption"] = "Не открывай дверь"
        self.write()
        receipt = produce(self.path, self.root / "burned.mp4")
        self.assertTrue(receipt["captions_burned_in"])
        self.assertTrue(receipt["automatic_music_ducking"])
        self.assertTrue((self.root / "burned.srt").is_file())
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_name,codec_type", "-of", "json", str(self.root / "burned.mp4")],
            capture_output=True, text=True, check=True,
        )
        codecs = {row["codec_name"] for row in json.loads(probe.stdout)["streams"]}
        self.assertIn("h264", codecs)
        self.assertIn("aac", codecs)
        self.assertGreater((self.root / "burned.mp4").stat().st_size, 10000)

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
