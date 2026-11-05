"""Offline tests only: no Kaggle API calls and no GPU quota consumed."""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from studio.comfy_kaggle import SLUG, build_job, reviewed_source, verify


class KaggleComfyTests(unittest.TestCase):
    def test_reviewed_bundle_contains_only_known_fixture_and_studio(self):
        data = reviewed_source()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
        self.assertIn("studio/art_director.py", names)
        self.assertIn("templates/comfyui_sdxl_api.json", names)
        self.assertIn("episodes/E01_THE_REFLECTION.json", names)
        self.assertFalse(any(".env" in name or ".github" in name for name in names))

    def test_private_submission_compiles_and_has_no_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            meta = build_job("examplekaggle", "comfy-123-1", output_dir=target, git_sha="a" * 40)
            kernel = json.loads((target / "kernel-metadata.json").read_text(encoding="utf-8"))
            script = (target / "kernel_main.py").read_text(encoding="utf-8")
            compile(script, "kernel_main.py", "exec")
            self.assertEqual(meta["kernel"], "examplekaggle/" + SLUG)
            self.assertEqual(kernel["is_private"], "true")
            self.assertEqual(kernel["enable_gpu"], "true")
            self.assertEqual(kernel["machine_shape"], "NvidiaTeslaT4")
            self.assertNotIn("__TW_COMFY_CONFIG__", script)
            self.assertNotIn("__TW_COMFY_SOURCE__", script)
            self.assertNotIn("KAGGLE_API_TOKEN", script)
            self.assertFalse(meta["credentials_embedded"])

    def test_invalid_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_job("../hacker", "comfy-safe-1", output_dir=Path(tmp))
            with self.assertRaises(ValueError):
                build_job("safeuser", "../../wrong", output_dir=Path(tmp))

    def test_receipt_check_proves_real_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            image = Image.effect_noise((640, 1152), 40).convert("RGB")
            data = io.BytesIO()
            image.save(data, "PNG")
            png = data.getvalue()
            prompt_id = "comfy-test-prompt-id"
            manifest = {
                "comfy_prompt_id": prompt_id,
                "checkpoint_file": "sd_xl_base_1.0.safetensors",
                "png_sha256": hashlib.sha256(png).hexdigest(),
                "review_status": "unreviewed",
            }
            (target / "two_worlds_job_receipt.json").write_text(
                json.dumps({"nonce": "comfy-123-1", "success": True,
                            "comfy_prompt_id": prompt_id}), encoding="utf-8"
            )
            with zipfile.ZipFile(target / "TWO_WORLDS_ARTIFACTS.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("render.png", png)
                archive.writestr("render_manifest.json", json.dumps(manifest))
            result = verify(target, "comfy-123-1")
            self.assertEqual((result["width"], result["height"]), (640, 1152))
            self.assertFalse(result["art_approved"])
            with self.assertRaisesRegex(ValueError, "nonce"):
                verify(target, "comfy-stale-1")
            manifest["png_sha256"] = "0" * 64
            with zipfile.ZipFile(target / "TWO_WORLDS_ARTIFACTS.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("render.png", png)
                archive.writestr("render_manifest.json", json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify(target, "comfy-123-1")

    def test_gpu_workflow_is_explicit_once_and_private(self):
        from studio.comfy_kaggle import ROOT
        wf = (ROOT / ".github/workflows/comfyui-kaggle.yml").read_text(encoding="utf-8")
        self.assertIn("ops/comfy_launch.trigger", wf)
        self.assertIn("workflow_dispatch:", wf)
        self.assertIn("secrets.KAGGLE_API_TOKEN", wf)
        self.assertNotIn("schedule:", wf)
        self.assertIn("github.event.repository.private", wf)
        self.assertNotIn("metricool", wf.lower())


if __name__ == "__main__":
    unittest.main()
