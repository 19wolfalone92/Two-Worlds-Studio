"""Synthetic BYTES only: never interpreted as actual Wan checkpoints."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("freevideo_weight_integrity", HERE / "weights_preflight.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

class WeightsPreflightTests(unittest.TestCase):
    def create_fixture(self, root):
        files = {}
        for name in module.REQUIRED:
            target = root / name
            target.write_bytes(("MOCKED FILE CONTENT " + name).encode())
            files[name] = {"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        manifest = root / "weights.json"
        doc = {"schema": 1, "model": "wan22-5b", "license_reviewed": True,
               "source_repository": "https://huggingface.co/Comfy-Org/",
               "files": files}
        manifest.write_text(json.dumps(doc))
        return manifest, doc

    def test_matching_mock_files_never_claim_authenticated_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path, doc = self.create_fixture(Path(temp))
            status = module.verify_weights_manifest(path)
            self.assertEqual(status["status"], "LOCAL_HASHES_MATCH_OPERATOR_MANIFEST")
            self.assertFalse(status["publisher_signature_verified"])
            self.assertFalse(status["real_gpu_tested"])
            self.assertEqual(len(status["components"]), 3)

    def test_modified_weight_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            path, doc = self.create_fixture(Path(temp))
            target = Path(doc["files"][next(iter(doc["files"]))]["path"])
            target.write_bytes(b"changed weight bytes")
            with self.assertRaisesRegex(module.IntegrityBlocked, "digest mismatch"):
                module.verify_weights_manifest(path)

    def test_symlink_weight_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, doc = self.create_fixture(root)
            name = next(iter(doc["files"]))
            target = root / name
            copy = root / "original_file"
            target.rename(copy)
            target.symlink_to(copy)
            with self.assertRaisesRegex(module.IntegrityBlocked, "symlinked"):
                module.verify_weights_manifest(path)

    def test_missing_model_file_and_unreviewed_license_block(self):
        with tempfile.TemporaryDirectory() as temp:
            path, doc = self.create_fixture(Path(temp))
            doc["license_reviewed"] = False
            path.write_text(json.dumps(doc))
            with self.assertRaises(module.IntegrityBlocked):
                module.verify_weights_manifest(path)
            doc["license_reviewed"] = True
            doc["files"].pop(next(iter(doc["files"])))
            path.write_text(json.dumps(doc))
            with self.assertRaises(module.IntegrityBlocked):
                module.verify_weights_manifest(path)

    def test_duplicate_json_key_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            path, doc = self.create_fixture(Path(temp))
            path.write_text(path.read_text().replace('"schema": 1,', '"schema": 1, "schema": 1,'))
            with self.assertRaisesRegex(module.IntegrityBlocked, "duplicate"):
                module.verify_weights_manifest(path)

if __name__ == "__main__":
    unittest.main()
