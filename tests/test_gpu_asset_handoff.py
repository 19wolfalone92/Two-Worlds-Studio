"""No-API private Kaggle PNG importer regression checks."""
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from studio.gpu_asset_handoff import inspect_archive, stage


class PrivateGPUHandoffTest(unittest.TestCase):
    def make_bundle(self, root):
        root = Path(root)
        image = Image.effect_noise((640, 1152), 40).convert("RGB")
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        blob = payload.getvalue()
        manifest = {
            "png_sha256": hashlib.sha256(blob).hexdigest(),
            "comfy_prompt_id": "comfy-actual-render-123",
            "review_status": "unreviewed",
            "no_publication": True,
            "source_sha": "9" * 40, "scene": "S01",
        }
        receipt = {"nonce": "comfy-123-1", "success": True,
                   "comfy_prompt_id": "comfy-actual-render-123"}
        zip_path = root / "TWO_WORLDS_ARTIFACTS.zip"
        receipt_path = root / "two_worlds_job_receipt.json"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("render.png", blob)
            z.writestr("render_manifest.json", json.dumps(manifest))
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return zip_path, receipt_path

    def test_staged_image_is_unreviewed_never_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path, receipt = self.make_bundle(root)
            result = stage(zip_path, receipt, nonce="comfy-123-1",
                           destination=root / "private_assets" / "S01")
            self.assertEqual(result["status"], "staged_unreviewed")
            self.assertFalse(result["published"])
            meta = json.loads((root / "private_assets" / "S01" / "art_review.json").read_text())
            self.assertFalse(meta["master_ready"])
            self.assertTrue(meta["art_review_required"])
            with self.assertRaises(FileExistsError):
                stage(zip_path, receipt, nonce="comfy-123-1",
                      destination=root / "private_assets" / "S01")

    def test_wrong_nonce_and_checksum_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path, receipt = self.make_bundle(root)
            with self.assertRaisesRegex(ValueError, "another run"):
                inspect_archive(zip_path, receipt, nonce="comfy-123-2")
            with zipfile.ZipFile(zip_path, "a") as z:
                z.writestr("unlisted.txt", "unknown")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                inspect_archive(zip_path, receipt, nonce="comfy-123-1")


if __name__ == "__main__":
    unittest.main()
