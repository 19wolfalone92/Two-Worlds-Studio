"""No real user media, text, or paid calls are used in these tests."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from studio.autonomous_batch import inspect_story, plan, execute, atomic_report, watch


class AutonomousBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.queue, self.out = self.root / "private_queue", self.root / "private_exports"
        self.queue.mkdir()
        self.story = self.queue / "13"
        self.story.mkdir()
        (self.story / "story.txt").write_text("Существующая история не переписывается.", encoding="utf-8")
        self.original = (self.story / "story.txt").read_bytes()
        self.project = {"schema": 1, "id": "work_13", "title": "Тест",
                        "language": "ru", "script_file": "story.txt",
                        "script_sha256": hashlib.sha256(self.original).hexdigest(),
                        "width": 1080, "height": 1920, "fps": 30,
                        "scenes": [{"id": n, "image": f"images/{n}.png",
                                    "voice": f"voices/{n}.wav", "subtitles": []}
                                   for n in range(1, 9)]}
        self.save()

    def save(self):
        (self.story / "project.json").write_text(
            json.dumps(self.project, ensure_ascii=False), encoding="utf-8")

    def complete_media(self):
        (self.story / "images").mkdir(exist_ok=True)
        (self.story / "voices").mkdir(exist_ok=True)
        for scene in self.project["scenes"]:
            Image.new("RGB", (200, 400), (120, 60, 180)).save(self.story / scene["image"])
            (self.story / scene["voice"]).write_bytes(b"artificial-test-audio-not-a-real-WAV")
            scene["subtitles"] = [{"start": 0, "end": 1, "text": "Строка"}]
        self.save()

    def fake_output(self, *, black=False, tamper=False):
        folder = self.out / "work_13"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "final.mp4").write_bytes(b"test-only-not-real-video")
        (folder / "subtitles.srt").write_text("test", encoding="utf-8")
        Image.new("RGB", (300, 500), (0, 0, 0) if black else (85, 110, 160)).save(folder / "cover.jpg")
        sha = hashlib.sha256((folder / "final.mp4").read_bytes()).hexdigest()
        (folder / "qa_report.json").write_text(json.dumps({
            "technical_pass": True, "full_decode_pass": True,
            "approved_for_publication": False, "first_frame_mean": 45,
            "black_intervals": []}))
        (folder / "receipt.json").write_text(json.dumps({
            "story_id": "work_13", "script_sha256": self.project["script_sha256"],
            "final_sha256": "0" * 64 if tamper else sha, "review_status": "REVIEW_REQUIRED",
            "approved_for_publication": False}))

    def test_five_other_stories_wait_for_private_source(self):
        report = plan(self.queue, self.out)
        self.assertEqual(report["expected_stories"], 6)
        self.assertEqual(report["counts"]["WAITING_FOR_PRIVATE_SOURCE"], 5)
        self.assertFalse(report["approved_for_publication"])

    def test_missing_assets_not_misreported_as_finished(self):
        one = inspect_story(self.queue, self.out, "13")
        self.assertEqual(one["status"], "ASSETS_PENDING")
        self.assertEqual(one["missing"], {"images": 8, "voices": 8, "subtitles": 8})
        self.assertEqual(one["next_action"], "GENERATE_IMAGES_PRIVATE_GPU")

    def test_original_narration_sha_must_match(self):
        (self.story / "story.txt").write_text("Подмена сюжета", encoding="utf-8")
        self.assertEqual(inspect_story(self.queue, self.out, "13")["status"], "INPUT_BLOCKED")

    def test_escape_asset_path_is_rejected(self):
        self.project["scenes"][0]["image"] = "../../somewhere.png"
        self.save()
        self.assertEqual(inspect_story(self.queue, self.out, "13")["status"], "INPUT_BLOCKED")

    def test_render_only_complete_stories_preserves_original(self):
        self.complete_media()
        with patch("studio.shorts_factory.validate_project"):
            self.assertEqual(inspect_story(self.queue, self.out, "13")["status"], "READY_TO_RENDER")
            with patch("studio.shorts_factory.run", return_value={"status": "REVIEW_REQUIRED"}) as renderer:
                report = execute(self.queue, self.out, max_jobs=1)
        renderer.assert_called_once()
        self.assertEqual(report["render_attempts"], [{"story": "13", "status": "REVIEW_REQUIRED"}])
        self.assertEqual((self.story / "story.txt").read_bytes(), self.original)
        self.assertFalse(report["approved_for_publication"])

    def test_valid_output_requires_review_not_publication(self):
        self.complete_media()
        self.fake_output()
        with patch("studio.shorts_factory.validate_project"):
            self.assertEqual(inspect_story(self.queue, self.out, "13")["status"], "REVIEW_REQUIRED")
            with patch("studio.shorts_factory.run") as renderer:
                execute(self.queue, self.out)
                renderer.assert_not_called()

    def test_black_cover_and_modified_file_block_output(self):
        self.complete_media()
        for issue in ("black", "tamper"):
            self.fake_output(black=issue == "black", tamper=issue == "tamper")
            with patch("studio.shorts_factory.validate_project"):
                self.assertEqual(inspect_story(self.queue, self.out, "13")["status"], "OUTPUT_BLOCKED")

    def test_watch_one_cycle_never_sleeps(self):
        with patch("studio.autonomous_batch.time.sleep") as sleeper:
            report = watch(self.queue, self.out, max_jobs=1, interval_seconds=30, max_cycles=1)
            sleeper.assert_not_called()
        self.assertEqual(report["technical_exports_ready_for_review"], 0)

    def test_status_json_does_not_expose_original_text(self):
        report = plan(self.queue, self.out)
        status = self.root / "status.json"
        atomic_report(status, report)
        payload = status.read_text(encoding="utf-8")
        self.assertNotIn("Существующая история", payload)
        self.assertEqual(json.loads(payload)["expected_stories"], 6)
