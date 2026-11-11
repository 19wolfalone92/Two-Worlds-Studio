"""No-download model installer contract: licence, paths and private disk isolation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from studio.model_install import COMPONENTS, _verify_checkpoint, install, inventory


class ModelInstallerTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.comfy = Path(folder.name) / "private-comfy"
        (self.comfy / "models").mkdir(parents=True)

    def test_model_inventory_is_offline_and_does_not_create_checkpoints(self):
        report = install("wan22-5b", self.comfy, execute=False, accept_license=False)
        self.assertEqual(report["status"], "OFFLINE_INVENTORY")
        self.assertEqual(len(report["files"]), 3)
        self.assertTrue(all(not row["installed"] for row in report["files"]))
        self.assertFalse(report["download_performed"])
        self.assertEqual(len(list(self.comfy.rglob("*.safetensors"))), 0)

    def test_ltx_components_are_pinned_to_official_hf_repos(self):
        self.assertEqual(len(COMPONENTS["ltx25-dfr"]), 6)
        info = inventory("ltx25-dfr", self.comfy)
        self.assertTrue(all(str(row["target"]).startswith(str(self.comfy / "models"))
                            for row in info["files"]))

    def test_explicit_acceptance_blocks_download_before_network(self):
        with self.assertRaises(PermissionError):
            install("wan22-5b", self.comfy, execute=True, accept_license=False)
        self.assertEqual(len(list(self.comfy.rglob("*.safetensors"))), 0)

    def test_same_size_but_wrong_sha_is_rejected(self):
        import hashlib
        model = self.comfy / "models" / "fake_checkpoint.safetensors"
        model.write_bytes(b"BAD" * 400)
        legitimate = hashlib.sha256(b"OK!" * 400).hexdigest()
        self.assertEqual(model.stat().st_size, 1200)
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            _verify_checkpoint(model, 1200, legitimate)
        with self.assertRaisesRegex(RuntimeError, "byte count"):
            _verify_checkpoint(model, 1199, None)
        with self.assertRaisesRegex(RuntimeError, "Invalid upstream"):
            _verify_checkpoint(model, 1200, "not-a-sha")
        _verify_checkpoint(model, 1200, hashlib.sha256(model.read_bytes()).hexdigest())

    def test_source_tree_does_not_accept_private_model_downloads(self):
        from studio.model_install import _safe_root
        from studio import model_install
        source = Path(model_install.__file__).resolve().parent.parent
        with self.assertRaises(ValueError):
            _safe_root(source)

    def test_symlink_as_destination_is_refused(self):
        target = self.comfy / "models" / "diffusion_models"
        target.mkdir()
        (target / "wan2.2_ti2v_5B_fp16.safetensors").symlink_to(self.comfy / "missing")
        with self.assertRaisesRegex(ValueError, "symlink"):
            inventory("wan22-5b", self.comfy)


if __name__ == "__main__":
    unittest.main()
