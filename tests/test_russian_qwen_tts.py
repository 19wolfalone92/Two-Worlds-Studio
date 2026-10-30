"""CPU/offline contract tests for Russian voice registration; no model downloads."""
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


TEXT_RU = "Когда часы пробили полночь, она услышала знакомый голос. " * 5


class RussianQwenVoiceTests(unittest.TestCase):
    def _files(self, root, *, language="ru", voice="qwen:Serena", style="sad"):
        root = Path(root)
        (root / "original.txt").write_text(TEXT_RU, encoding="utf-8")
        manifest = root / "private_jobs.json"
        manifest.write_text(json.dumps([{
            "id": "new_story_01", "text_file": "original.txt",
            "language": language, "voice": voice, "style": style,
        }], ensure_ascii=False), encoding="utf-8")
        return manifest

    def test_registers_russian_voice_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = self._files(root)
            before = (root / "original.txt").read_bytes()
            jobs = prepare(manifest, root / "output")
            self.assertEqual(jobs[0]["engine"], "qwen3-tts")
            self.assertEqual(jobs[0]["language"], "ru")
            self.assertEqual(jobs[0]["voice"], "qwen:Serena")
            self.assertEqual(jobs[0]["style"], "sad")
            self.assertEqual(jobs[0]["model"], MODEL_SMALL)
            self.assertEqual(jobs[0]["text_sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual((root / "original.txt").read_bytes(), before)

    def test_dry_run_never_imports_model_or_makes_audio(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jobs = prepare(self._files(root), root / "out")
            with patch("studio.narration_batch.russian_synthesize") as tts:
                result = execute(jobs, root / "out", dry_run=True)
                tts.assert_not_called()
            self.assertEqual(result["actual_qwen_wavs"], 0)
            self.assertEqual(result["actual_kokoro_wavs"], 0)
            self.assertNotIn("text", result["jobs"][0])
            self.assertFalse((root / "out").exists())

    def test_russian_dispatch_and_receipt_are_unreviewed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jobs = prepare(self._files(root), root / "out")
            def mocked(text, voice, path, *, style, model_id):
                self.assertEqual(text, TEXT_RU.strip())
                self.assertEqual((voice, style, model_id), ("Serena", "sad", MODEL_SMALL))
                path.write_bytes(b"placeholder-test-data-not-a-real-wav")
                return {"path": str(path), "review_status": "unreviewed"}
            with patch("studio.narration_batch.russian_synthesize", side_effect=mocked):
                result = execute(jobs, root / "out")
            self.assertEqual(result["actual_qwen_wavs"], 1)
            self.assertEqual(result["actual_kokoro_wavs"], 0)
            receipt = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertFalse(receipt["human_review"])
            self.assertEqual(receipt["wav_tracks"][0]["language"], "ru")
            self.assertEqual(receipt["wav_tracks"][0]["review_status"], "unreviewed")

    def test_reject_english_text_with_russian_voice(self):
        with self.assertRaises(ValueError):
            validate("Just an English sentence.", "Serena", "warm", MODEL_SMALL)

    def test_reject_unknown_voice_style_model(self):
        for voice, style, model in [
            ("evilvoice", "sad", MODEL_SMALL),
            ("Serena", "panic_attack", MODEL_SMALL),
            ("Serena", "sad", "some/unapproved-model"),
        ]:
            with self.subTest(voice=voice, style=style, model=model):
                with self.assertRaises(ValueError):
                    validate(TEXT_RU, voice, style, model)

    def test_reject_speaker_language_mix(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = self._files(root, language="a")
            with self.assertRaises(ValueError):
                prepare(manifest, root / "out")

    def test_reject_path_escape_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = self._files(root)
            obj = json.loads(manifest.read_text(encoding="utf-8"))
            obj[0]["text_file"] = "../private.txt"
            manifest.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare(manifest, root / "out")
            obj[0]["text_file"] = "symlink.txt"
            (root / "symlink.txt").symlink_to(Path(d).parent / "outside.txt")
            manifest.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare(manifest, root / "out")

    def test_split_preserves_words_and_chunk_limit(self):
        chunks = split_text(TEXT_RU, limit=100)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(x) <= 100 for x in chunks))
        self.assertEqual(" ".join(chunks), " ".join(TEXT_RU.split()))

    def test_split_rejects_runaway_word(self):
        with self.assertRaises(ValueError):
            split_text("Ы" * 400, limit=100)

    def test_russian_voice_catalog_and_styles(self):
        self.assertIn("Serena", VOICES)
        self.assertIn("tense", STYLE_PROMPTS)
        self.assertIn("warm", STYLE_PROMPTS)


if __name__ == "__main__":
    unittest.main()
