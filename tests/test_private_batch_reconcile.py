"""Synthetic ZIP fixtures only, no private stories, GPU or published video."""
import hashlib
import io
import json
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

from PIL import Image
from studio.private_batch_reconcile import ReconcileBlocked, reconcile


def sha(data):
    return hashlib.sha256(data).hexdigest()


class PrivateBatchReconcileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.media = self.root / "media"
        self.media.mkdir()
        for n in range(13, 19):
            folder = self.source / "production_queue" / str(n)
            folder.mkdir(parents=True)
            content = ("Original test script %s." % n).encode()
            (folder / "story.txt").write_bytes(content)
            scenes = [{"id": j, "image": "images/scene_%02d.png" % j,
                       "voice": "voices/scene_%02d.wav" % j, "subtitles": []}
                      for j in range(1, 9)]
            project = {"schema": 1, "id": "work_%s" % n, "language": "ru",
                       "title": "Тест №%s" % n, "width": 1080, "height": 1920,
                       "fps": 24, "script_file": "story.txt",
                       "script_sha256": sha(content), "scenes": scenes}
            (folder / "project.json").write_text(json.dumps(project))

    def make(self, phase, *, bad_checksum=False, hostile=False):
        folder = self.source / "production_queue" / "13"
        original = json.loads((folder / "project.json").read_text())
        archived = json.loads(json.dumps(original))
        files = {"project.json": json.dumps(archived).encode(),
                 "story.txt": (folder / "story.txt").read_bytes()}
        for j in range(1, 9):
            if phase == "image":
                image = Image.new("RGB", (200, 360), (60, 130, 90))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                files["images/scene_%02d.png" % j] = buffer.getvalue()
            else:
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as w:
                    w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
                    w.writeframes(b"\x00\x01" * 9600)
                files["voices/scene_%02d.wav" % j] = buffer.getvalue()
                archived["scenes"][j-1]["subtitles"] = [
                    {"start": .05, "end": .5, "text": "Тест"}]
        files["project.json"] = json.dumps(archived).encode()
        if hostile:
            files["../other_private_scene.txt"] = b"do not extract"
        inventory = []
        archive = self.media / ("TwoWorlds_PRIVATE_13_%s.zip" % phase)
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path, data in files.items():
                arc = "production_queue/13/" + path
                z.writestr(arc, data)
                inventory.append({"path": arc, "sha256": sha(data), "bytes": len(data)})
            if bad_checksum:
                inventory[0]["sha256"] = "0" * 64
            z.writestr("run_manifest.json", json.dumps({
                "schema": 1, "story": "13", "script_sha256": original["script_sha256"],
                "generated_media": inventory, "reviewed": False, "published": False}))
        return archive

    def test_join_real_assets_without_editing_locked_source(self):
        source_before = (self.source / "production_queue/13/story.txt").read_bytes()
        self.make("image")
        self.make("voice")
        result = reconcile(self.source, self.media, self.root / "queue")
        self.assertEqual(result["source_stories"], 6)
        self.assertEqual(result["production"]["counts"]["READY_TO_RENDER"], 1)
        self.assertEqual(result["production"]["counts"]["ASSETS_PENDING"], 5)
        self.assertFalse(result["published"])
        self.assertEqual((self.root / "queue/13/story.txt").read_bytes(), source_before)
        self.assertEqual((self.source / "production_queue/13/story.txt").read_bytes(), source_before)
        with self.assertRaises(ReconcileBlocked):
            reconcile(self.source, self.media, self.root / "queue")

    def test_sha_tampering_blocks_atomic_output(self):
        self.make("image", bad_checksum=True)
        with self.assertRaises(ReconcileBlocked):
            reconcile(self.source, self.media, self.root / "queue")
        self.assertFalse((self.root / "queue").exists())

    def test_zip_escape_blocks(self):
        self.make("voice", hostile=True)
        with self.assertRaises(ReconcileBlocked):
            reconcile(self.source, self.media, self.root / "queue")
        self.assertFalse((self.root / "queue").exists())

    def test_without_media_accurately_reports_missing(self):
        result = reconcile(self.source, self.media, self.root / "queue")
        self.assertEqual(result["production"]["counts"]["ASSETS_PENDING"], 6)
        self.assertEqual(result["production"]["technical_exports_ready_for_review"], 0)
