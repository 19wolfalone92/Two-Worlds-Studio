"""CPU-only tests: no Kaggle request, no neural video generation."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_private_job as module


class KagglePrivateJobTests(unittest.TestCase):
    def args(self, root: Path):
        source = root / "private_photo.png"
        Image.new("RGB", (600, 800), (55, 66, 77)).save(source)
        return {
            "username": "privateaccount", "nonce": "freevideo-test-20261123",
            "source": source, "prompt": (
                "The man takes a natural breath and gently turns his head in one continuous shot."),
            "destination": root / "kernel-private",
            "revision": "a" * 40, "accept_model_license": True
        }

    def test_prepared_private_kernel_contains_only_embedded_photo_not_public_git(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.args(root)
            source_hash = module.sha256(args["source"])
            result = module.prepare_private_kernel(**args)
            self.assertEqual(result["status"], "PRIVATE_KAGGLE_PACKAGE_PREPARED_NO_GPU")
            self.assertFalse(result["gpu_executed"])
            self.assertFalse(result["published"])
            self.assertTrue(result["source_private"])
            self.assertEqual(module.sha256(args["source"]), source_hash)
            metadata = json.loads((args["destination"] / "kernel-metadata.json").read_text())
            self.assertEqual(metadata["is_private"], "true")
            self.assertEqual(metadata["enable_gpu"], "true")
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            worker = (args["destination"] / "kernel_main.py").read_text()
            self.assertNotIn("__FREEVIDEO_JOB_CONFIG_B64__", worker)
            self.assertNotIn("__FREEVIDEO_PRIVATE_IMAGE_B64__", worker)
            self.assertNotIn("KAGGLE_API_TOKEN", worker)
            self.assertNotIn("HF_TOKEN=", worker)
            self.assertIn("studio.comfy_neural/wan22-5b", worker)
            self.assertIn(base64.b64encode(args["source"].read_bytes()).decode(), worker)
            self.assertFalse((args["destination"] / "private_photo.png").exists())
            with self.assertRaisesRegex(module.KaggleBlocked, "NEW private"):
                module.prepare_private_kernel(**args)

    def test_license_refusal_has_no_files_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            args["accept_model_license"] = False
            with self.assertRaisesRegex(module.KaggleBlocked, "licence"):
                module.prepare_private_kernel(**args)
            self.assertFalse(args["destination"].exists())

    def test_unknown_git_sha_is_rejected_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            args["revision"] = "main"
            with self.assertRaisesRegex(module.KaggleBlocked, "40-hex"):
                module.prepare_private_kernel(**args)
            self.assertFalse(args["destination"].exists())

    def test_image_larger_than_limit_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            args["source"].write_bytes(b"0" * (8 * 1024 * 1024 + 1))
            with self.assertRaisesRegex(module.KaggleBlocked, "8 MiB"):
                module.prepare_private_kernel(**args)
            self.assertFalse(args["destination"].exists())

    def test_photo_always_remains_outside_public_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            args["source"] = Path(__file__)
            with self.assertRaises(module.KaggleBlocked):
                module.prepare_private_kernel(**args)

    def test_missing_result_does_not_claim_gpu_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((FileNotFoundError, ValueError)):
                module.verify_private_result(Path(tmp), "freevideo-test-20261123")

    def test_fake_zip_video_rejected_even_when_hashes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nonce = "freevideo-test-20261123"
            arc = root / "FREEVIDEO_PRIVATE_PREVIEW.zip"
            payload = b"not a video, only mocked test bytes" * 1000
            digest = hashlib.sha256(payload).hexdigest()
            with zipfile.ZipFile(arc, "w") as z:
                z.writestr("preview.mp4", payload)
                z.writestr("preview.receipt.json", json.dumps({
                    "clip_sha256": digest, "inference_performed": True}))
                z.writestr("preview.freevideo.json", json.dumps({"clip_sha256": digest}))
                z.writestr("preview.hybrid.json", json.dumps({
                    "clip_sha256": digest, "stage": "preview"}))
                z.writestr("job.json", "{}")
            proof = {
                "nonce": nonce, "status": "REAL_GPU_PREVIEW_REVIEW_REQUIRED",
                "human_approved": False, "published": False,
                "visual_quality": "NOT_VERIFIED", "clip_sha256": digest,
                "archive_sha256": module.sha256(arc)}
            (root / "freevideo_result.json").write_text(json.dumps(proof))
            # Even matching fake bytes/hashes cannot be accepted as real
            # ComfyUI provenance or full-frame decoded AI video.
            with self.assertRaisesRegex(module.KaggleBlocked, "provenance mismatch"):
                module.verify_private_result(root, nonce)

    def test_modified_kaggle_zip_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nonce = "freevideo-test-20261123"
            archive = root / "FREEVIDEO_PRIVATE_PREVIEW.zip"
            archive.write_bytes(b"0" * 10000)
            (root / "freevideo_result.json").write_text(json.dumps({
                "nonce": nonce, "status": "REAL_GPU_PREVIEW_REVIEW_REQUIRED",
                "human_approved": False, "published": False,
                "visual_quality": "NOT_VERIFIED", "archive_sha256": "f" * 64}))
            with self.assertRaisesRegex(module.KaggleBlocked, "ZIP SHA"):
                module.verify_private_result(root, nonce)


if __name__ == "__main__":
    unittest.main()
