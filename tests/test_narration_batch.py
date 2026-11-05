"""No-model unit tests for batch narration and private script handling."""
import json
import tempfile
import unittest
from pathlib import Path
from studio.narration_batch import prepare, execute


class NarrationBatchTests(unittest.TestCase):
    def test_dry_run_does_not_require_kokoro_or_generate_audio(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "story01.txt").write_text("A simple mysterious story. " * 12, encoding="utf-8")
            manifest = root / "stories.json"
            manifest.write_text(json.dumps([{
                "id": "story01", "text_file": "story01.txt",
                "voice": "bm_george", "language": "b"
            }]), encoding="utf-8")
            jobs = prepare(manifest, root / "out")
            result = execute(jobs, root / "out", dry_run=True)
            self.assertEqual(result["actual_kokoro_wavs"], 0)
            self.assertEqual(len(result["jobs"]), 1)
            self.assertNotIn("text", result["jobs"][0])
            self.assertFalse((root / "out").exists())

    def test_deny_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = root / "stories.json"
            manifest.write_text(json.dumps([{
                "id": "story01", "text_file": "../secret.txt",
                "voice": "af_heart", "language": "a"
            }]), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare(manifest, root / "out")

    def test_reject_wrong_language(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "story.txt").write_text("Mystery story. " * 10, encoding="utf-8")
            manifest = root / "stories.json"
            manifest.write_text(json.dumps([{
                "id": "story01", "text_file": "story.txt",
                "voice": "bm_george", "language": "a"
            }]), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare(manifest, root / "out")
