"""CPU-only Russian voice contract tests. No model weights are downloaded."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.narration_batch import prepare, execute
from studio.russian_qwen_tts import (
    MODEL_SMALL, STYLE_PROMPTS, VOICES, split_text, validate,
)

RUSSIAN = "Когда часы пробили полночь, она услышала знакомый голос. " * 5


class RussianVoiceTests(unittest.TestCase):
    def make_job(self, root, *, language="ru", voice="qwen:Serena", style="sad"):
        root = Path(root)
        (root / "original.txt").write_text(RUSSIAN, encoding="utf-8")
        manifest = root / "jobs.json"
        manifest.write_text(json.dumps([{
            "id": "story_01", "text_file": "original.txt",
            "language": language, "voice": voice, "style": style
        }], ensure_ascii=False), encoding="utf-8")
        return manifest

    def test_russian_preparation_preserves_original_script(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = self.make_job(root)
            before = (root / "original.txt").read_bytes()
            jobs = prepare(manifest, root / "out")
            self.assertEqual(jobs[0]["engine"], "qwen3-tts")
            self.assertEqual(jobs[0]["voice"], "qwen:Serena")
            self.assertEqual(jobs[0]["style"], "sad")
            self.assertEqual(jobs[0]["model"], MODEL_SMALL)
            self.assertEqual(jobs[0]["text_sha256"],
                             hashlib.sha256(RUSSIAN.strip().encode()).hexdigest())
            self.assertEqual((root / "original.txt").read_bytes(), before)

    def test_dry_run_no_tts_no_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jobs = prepare(self.make_job(root), root / "out")
            with patch("studio.narration_batch.russian_synthesize") as engine:
                response = execute(jobs, root / "out", dry_run=True)
                engine.assert_not_called()
            self.assertEqual(response["actual_qwen_wavs"], 0)
            self.assertNotIn("text", response["jobs"][0])
            self.assertFalse((root / "out").exists())

    def test_dispatch_and_unreviewed_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jobs = prepare(self.make_job(root), root / "out")
            def stub(text, voice, path, *, style, model_id):
                self.assertEqual((voice, style, model_id), ("Serena", "sad", MODEL_SMALL))
                path.write_bytes(b"mock-audio-only")
                return {"path": str(path), "review_status": "unreviewed"}
            with patch("studio.narration_batch.russian_synthesize", side_effect=stub):
                result = execute(jobs, root / "out")
            self.assertEqual(result["actual_qwen_wavs"], 1)
            receipt = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertFalse(receipt["human_review"])
            self.assertEqual(receipt["wav_tracks"][0]["language"], "ru")
            self.assertEqual(receipt["wav_tracks"][0]["review_status"], "unreviewed")

    def test_rejects_english_or_invalid_voice_and_style(self):
        cases = [
            ("English text.", "Serena", "warm", MODEL_SMALL),
            (RUSSIAN, "missing", "sad", MODEL_SMALL),
            (RUSSIAN, "Serena", "unknown", MODEL_SMALL),
            (RUSSIAN, "Serena", "sad", "unknown/model"),
        ]
        for item in cases:
            with self.subTest(item=item), self.assertRaises(ValueError):
                validate(*item)

    def test_rejects_english_mapping_and_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = self.make_job(root, language="a")
            with self.assertRaises(ValueError):
                prepare(manifest, root / "out")
            manifest = self.make_job(root)
            job = json.loads(manifest.read_text(encoding="utf-8"))[0]
            job["text_file"] = "../secret.txt"
            manifest.write_text(json.dumps([job]), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare(manifest, root / "out")

    def test_chunks_preserve_words_with_limit(self):
        chunks = split_text(RUSSIAN, limit=100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(x) <= 100 for x in chunks))
        self.assertEqual(" ".join(chunks), " ".join(RUSSIAN.split()))
        with self.assertRaises(ValueError):
            split_text("Ы" * 400, limit=100)

    def test_catalog_contains_emotional_options(self):
        self.assertIn("Serena", VOICES)
        self.assertIn("tense", STYLE_PROMPTS)
        self.assertIn("warm", STYLE_PROMPTS)


if __name__ == "__main__":
    unittest.main()
