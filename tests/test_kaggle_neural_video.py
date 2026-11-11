"""Offline acceptance contract for a single Kaggle GPU experiment."""
from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from studio.kaggle_neural_video import FRAME_COUNT, HEIGHT, WIDTH, prepare, verify


class NeuralKaggleGPUProofTests(unittest.TestCase):
    def test_private_job_has_one_17_frame_wan5b_job_and_no_secrets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "kernel"
            result = prepare("test-account", "dryrun-20261111", target, revision="a"*40)
            self.assertEqual(result["frames"], FRAME_COUNT)
            self.assertEqual(result["size"], [WIDTH, HEIGHT])
            self.assertFalse(result["gpu_executed"])
            self.assertFalse(result["credentials_included"])
            meta = json.loads((target / "kernel-metadata.json").read_text())
            self.assertEqual(meta["is_private"], "true")
            self.assertEqual(meta["enable_gpu"], "true")
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(meta["dataset_sources"], [])
            script = (target / "kernel_main.py").read_text()
            self.assertNotIn("KAGGLE_API_TOKEN", script)
            self.assertNotIn("HF_TOKEN=", script)
            self.assertNotIn("__TW_VIDEO_CONFIG__", script)
            self.assertNotIn("__TW_VIDEO_BUNDLE__", script)
            self.assertIn("wan22-5b", script)

    def test_invalid_job_name_does_not_write_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            dst = Path(folder) / "kernel"
            with self.assertRaises(ValueError):
                prepare("../untrusted", "dryrun-20261111", dst)
            self.assertFalse(dst.exists())

    def test_verify_fails_closed_without_real_media(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            receipt = {"nonce": "dryrun-20261111", "success": True,
                       "model": "wan22-5b", "inference_performed": True,
                       "video": {"frames": FRAME_COUNT, "width": WIDTH, "height": HEIGHT},
                       "clip_sha256": "a"*64}
            (root / "two_worlds_job_receipt.json").write_text(json.dumps(receipt))
            archive = root / "TWO_WORLDS_ARTIFACTS.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("neural_proof.mp4", b"\x00\x00xxftyp" + b"0" * 9000)
                z.writestr("neural_proof.receipt.json", json.dumps(receipt))
                z.writestr("technical_qc.json", json.dumps({
                    "all_frames_decoded": True, "real_motion_detected": True
                }))
            with self.assertRaises(ValueError):
                verify(root, "dryrun-20261111")


if __name__ == "__main__":
    unittest.main()
