"""Offline tests for Russian shorts: sample WAV is artificial, not a released voice."""
import hashlib
import json
import math
import shutil
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


if __name__ == "__main__":
    unittest.main()
