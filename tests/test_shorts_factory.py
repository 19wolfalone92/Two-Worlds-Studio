"""Offline tests for Russian shorts: sample WAV is artificial, not a released voice."""
import hashlib
import json
import math
import shutil
import subprocess
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from PIL import Image
from studio.shorts_factory import Blocked, run, validate_project


class ShortsFactoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        (self.root / "audio").mkdir()
        for n, rgb in ((1, (67, 100, 130)), (2, (150, 85, 60))):
            Image.new("RGB", (400, 720), rgb).save(self.root / f"images/{n}.png")
            with wave.open(str(self.root / f"audio/{n}.wav"), "wb") as voice:
                voice.setnchannels(1)
                voice.setsampwidth(2)
                voice.setframerate(12000)
                voice.writeframes(b"".join(
                    struct.pack("<h", int(2500 * math.sin(t * .1)))
                    for t in range(7200)))
        self.script = self.root / "story.txt"
        self.script.write_text("Оригинальный тестовый текст без изменений.", encoding="utf-8")
        self.doc = {
            "schema": 1, "id": "smoke01", "title": "Техническая проверка",
            "language": "ru", "script_file": "story.txt",
            "script_sha256": hashlib.sha256(self.script.read_bytes()).hexdigest(),
            "width": 192, "height": 336, "fps": 12,
            "scenes": [
                {"id": n, "image": f"images/{n}.png", "voice": f"audio/{n}.wav",
                 "subtitles": [{"start": .05, "end": .5, "text": "Проверка один два."}]}
                for n in (1, 2)
            ],
        }
        self.manifest = self.root / "project.json"
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.doc, ensure_ascii=False), encoding="utf-8")

    def test_dry_run_validates_without_output(self):
        result = run(self.manifest, self.root / "out", dry_run=True, smoke=True)
        self.assertEqual(result["status"], "ASSETS_READY")
        self.assertEqual(result["scenes"], 2)
        self.assertFalse((self.root / "out").exists())

    def test_master_requires_eight_scenes(self):
        with self.assertRaises(Blocked):
            validate_project(self.manifest)

    def test_missing_voice_fails_closed(self):
        self.doc["scenes"][1]["voice"] = "audio/not_found.wav"
        self.save()
        with self.assertRaisesRegex(Blocked, "Missing"):
            run(self.manifest, self.root / "out", smoke=True)
        self.assertFalse((self.root / "out").exists())

    def test_wrong_script_hash_is_rejected(self):
        self.doc["script_sha256"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(Blocked, "SHA-256"):
            run(self.manifest, self.root / "out", smoke=True)

    def test_deny_escaped_media_path(self):
        self.doc["scenes"][0]["image"] = "../outside.jpg"
        self.save()
        with self.assertRaises(Blocked):
            run(self.manifest, self.root / "out", smoke=True)

    def test_cues_must_fit_actual_audio(self):
        self.doc["scenes"][0]["subtitles"][0]["end"] = 4
        self.save()
        with self.assertRaisesRegex(Blocked, "timestamp"):
            run(self.manifest, self.root / "out", smoke=True)

    def test_cues_are_required(self):
        self.doc["scenes"][0]["subtitles"] = []
        self.save()
        with self.assertRaisesRegex(Blocked, "subtitles"):
            run(self.manifest, self.root / "out", smoke=True)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg absent")
    def test_real_mp4_is_generated_and_not_overwritten(self):
        before = self.script.read_bytes()
        result = run(self.manifest, self.root / "out", smoke=True)
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertTrue(result["qa"]["technical_pass"])
        self.assertFalse(result["qa"]["approved_for_publication"])
        self.assertTrue(all(Path(p).is_file() for p in result["files"]))
        self.assertEqual(self.script.read_bytes(), before)
        self.assertIn("Проверка", (self.root / "out/smoke01/subtitles.srt").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(Blocked, "Output exists"):
            run(self.manifest, self.root / "out", smoke=True)


    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg absent")
    def test_voice_triggers_music_ducking_and_output_stays_unapproved(self):
        shutil.copyfile(self.root / "audio/1.wav", self.root / "audio/music.wav")
        for scene in self.doc["scenes"]:
            scene["music"] = "audio/music.wav"
        self.save()
        result = run(self.manifest, self.root / "mixed", smoke=True)
        self.assertTrue(result["qa"]["music_for_all_scenes"])
        self.assertFalse(result["qa"]["approved_for_publication"])
        self.assertEqual(result["status"], "REVIEW_REQUIRED")


    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg absent")
    def test_real_motion_video_and_short_clip_rejection(self):
        path = self.root / "motion.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                        "-i", "testsrc2=size=192x336:rate=12",
                        "-t", "1", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(path)], check=True)
        self.doc["scenes"][0]["motion_video"] = "motion.mp4"
        self.save()
        result = run(self.manifest, self.root / "motion_out", smoke=True)
        self.assertEqual(result["qa"]["real_motion_video_scenes"], 1)
        self.assertTrue(result["qa"]["full_decode_pass"])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                        "-i", "testsrc2=size=192x336:rate=12",
                        "-t", "0.25", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(path)], check=True)
        with self.assertRaisesRegex(Blocked, "shorter than Russian narration"):
            validate_project(self.manifest, smoke=True)


    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg absent")
    def test_burned_subtitles_visible_in_lower_safe_zone(self):
        result = run(self.manifest, self.root / "captioned", smoke=True)
        frame = self.root / "caption-frame.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.30",
                        "-i", result["files"][0], "-frames:v", "1",
                        str(frame)], check=True)
        with Image.open(frame) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            whites_lower = 0
            for y in range(int(h * .52), int(h * .88)):
                for x in range(w):
                    r, g, b = rgb.getpixel((x, y))
                    whites_lower += (r > 225 and g > 225 and b > 225)
            self.assertGreater(whites_lower, 20, "No subtitle in lower safe zone")


if __name__ == "__main__":
    unittest.main()
