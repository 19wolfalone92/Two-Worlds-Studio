"""No-download model installer contract: licence, paths and private disk isolation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from studio.model_install import COMPONENTS, install, inventory


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

    def test_symlink_as_destination_is_refused(self):
        target = self.comfy / "models" / "diffusion_models"
        target.mkdir()
        (target / "wan2.2_ti2v_5B_fp16.safetensors").symlink_to(self.comfy / "missing")
        with self.assertRaisesRegex(ValueError, "symlink"):
            inventory("wan22-5b", self.comfy)


if __name__ == "__main__":
    unittest.main()
