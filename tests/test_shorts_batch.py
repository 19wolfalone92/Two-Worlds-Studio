"""Offline tests for Work story import; no private original scripts committed."""
import json
import tempfile
import unittest
from pathlib import Path

from studio.shorts_batch import create


class StoryImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.kit = self.base / "kit"
        (self.kit / "stories_ru").mkdir(parents=True)
        (self.kit / "narration_ru").mkdir()
        self.original = ("Русская история. " * 50).encode("utf-8")
        (self.kit / "narration_ru/13_sample.txt").write_bytes(self.original)
        scene = {"id": 1, "text": "Русский исходный текст сцены",
                 "visual_prompt_ru": "Свет в мастерской", "music_cue": "grief"}
        story = {"key": "13_demo", "lang": "ru", "title": "Тест истории",
                 "scenes": [dict(scene, id=i) for i in range(1, 9)]}
        (self.kit / "stories_ru/13.json").write_text(
            json.dumps(story, ensure_ascii=False), encoding="utf-8")

    def test_queue_tracks_missing_media_and_immutable_original(self):
        report = create(self.kit, self.base / "out")
        self.assertEqual(report["status"], "ASSETS_PENDING")
        self.assertEqual(report["jobs"][0]["scenes"], 8)
        self.assertEqual(report["jobs"][0]["missing"]["voices"], 8)
        self.assertEqual((self.base / "out/13/story.txt").read_bytes(), self.original)
        project = json.loads((self.base / "out/13/project.json").read_text(encoding="utf-8"))
        self.assertEqual(project["language"], "ru")
        self.assertEqual(project["scenes"][0]["subtitles"], [])
        self.assertFalse((self.base / "out/13/images").exists())
        with self.assertRaises(FileExistsError):
            create(self.kit, self.base / "out")

    def test_rejects_missing_original_voice_text(self):
        (self.kit / "narration_ru/13_sample.txt").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            create(self.kit, self.base / "out")


if __name__ == "__main__":
    unittest.main()
