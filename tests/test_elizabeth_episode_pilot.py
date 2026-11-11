"""Offline tests: Elizabeth EP01 is actually assembled by TWO WORLDS Studio.

No Kaggle GPU, paid services, repository secrets or public media are used.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

from PIL import Image, ImageDraw

from studio.elizabeth_episode_pilot import (
    SCENES, IMAGE_MODEL, PACK, RECEIPT, build_manifest, bundle, prepare,
    verify_outputs
)
from studio.hybrid_master import check_manifest


class ElizabethStudioPilotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _fill_test_inputs(self):
        art = self.root / "art"
        voice = self.root / "voices"
        art.mkdir()
        voice.mkdir()
        clocks = {}
        for idx, scene in enumerate(SCENES):
            name, speaker = scene[:2]
            pic = Image.new("RGB", (576, 1024), (20 + idx * 18, 29, 47))
            draw = ImageDraw.Draw(pic)
            for j in range(90):
                draw.rectangle(((j * 23) % 576, (j * 71) % 1024,
                                min(575, ((j * 23) % 576) + 5),
                                min(1023, ((j * 71) % 1024) + 6)),
                               fill=(j * 5 % 255, j * 3 % 255, j * 11 % 255))
            pic.save(art / (name + ".png"))
            if speaker:
                clocks[name] = 1.35 + idx / 10
                with wave.open(str(voice / (name + ".wav")), "wb") as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(24000)
                    f.writeframes(b"\0\0" * round(clocks[name] * 24000))
        with wave.open(str(self.root / "score.wav"), "wb") as f:
            f.setnchannels(2)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(b"\0\0\0\0" * 24000)
        return clocks

    def test_native_studio_manifest_has_seven_distinct_shots_six_locked_russian_voices(self):
        clock = self._fill_test_inputs()
        plan = build_manifest(self.root, clock)
        self.assertEqual(plan["schema"], 1)
        self.assertEqual([plan[x] for x in ("width", "height", "fps")], [576, 1024, 24])
        self.assertTrue(plan["burn_subtitles"])
        self.assertEqual(len(plan["shots"]), 7)
        self.assertFalse(plan["editorial_approved"])
        self.assertFalse(plan["rights_approved"])
        self.assertEqual(len({s["image"] for s in plan["shots"]}), 7)
        self.assertEqual(sum("voice" in s for s in plan["shots"]), 6)
        self.assertTrue(all(s.get("dialogue_text") == s.get("caption") for s in plan["shots"][1:]))
        normalized = check_manifest(plan, self.root, master=False)
        self.assertEqual(len(normalized["shots"]), 7)
        self.assertFalse(normalized["approved"])
        self.assertFalse(normalized["upgraded"])  # benign static camera preview
        self.assertEqual(normalized["language"], "ru")
        self.assertGreater(normalized["duration"], 17)

    def test_no_fake_speech_or_missing_images_allowed(self):
        clock = self._fill_test_inputs()
        (self.root / "voices" / "S03.wav").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Qwen voice"):
            build_manifest(self.root, clock)
        (self.root / "art" / "S02.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "GPU art"):
            build_manifest(self.root, clock)

    def test_unbounded_bad_voice_clock_rejected(self):
        clock = self._fill_test_inputs()
        clock["S02"] = 280.0
        with self.assertRaisesRegex(ValueError, "Natural voice"):
            build_manifest(self.root, clock)

    def test_private_gpu_job_packaging_contains_real_studio_modules(self):
        blob = bundle()
        self.assertGreater(len(blob), 10_000)
        result = prepare("wolf19loner92", "pilot-test-0001", self.root / "job")
        self.assertEqual(result["shots"], 7)
        self.assertFalse(result["real_gpu_execution"])
        self.assertTrue(result["private_review_only"])
        meta = json.loads((self.root / "job" / "kernel-metadata.json").read_text())
        self.assertEqual(meta["title"], meta["id"].split("/")[1])
        self.assertEqual(meta["is_private"], "true")
        self.assertEqual(meta["enable_gpu"], "true")
        script = (self.root / "job" / "kernel_main.py").read_text()
        self.assertNotIn("__ELIZABETH_CONFIG_B64__", script)
        self.assertNotIn("__ELIZABETH_SOURCE_B64__", script)
        self.assertNotIn("sk-", script)

    def test_forged_receipt_is_not_a_movie(self):
        self.assertNotIn("tiktok", IMAGE_MODEL.lower())
        (self.root / RECEIPT).write_text(json.dumps({
            "schema": 1, "run_nonce": "test-bad", "success": True,
            "provider": "native_two_worlds_studio",
            "pipeline": "studio.hybrid_master.produce",
            "shots": 7, "voices": 6,
            "published": False, "editorial_approved": False
        }))
        (self.root / PACK).write_bytes(b"not an MP4 or valid ZIP")
        with self.assertRaises(ValueError):
            verify_outputs(self.root, "test-bad")


if __name__ == "__main__":
    unittest.main()
