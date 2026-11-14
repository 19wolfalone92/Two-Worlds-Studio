"""Offline contracts for the owner-operated Kaggle hardware probe."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.verify_compute_preflight import EXPECTED_CPU_SHA, validate_receipt


def sample():
    return {
        "schema": 1,
        "source": "owner_operated_kaggle_notebook",
        "python": "3.11.0",
        "cpu": {"logical_cores": 4, "ram_gib": 14.7, "disk_free_gib": 12.4,
                "smoke": "PASS", "smoke_sha256": EXPECTED_CPU_SHA, "smoke_ms": 1},
        "gpu": {"torch_installed": True, "cuda_available": True,
                "device_count": 1,
                "devices": [{"name": "Tesla T4", "vram_gib": 15.0}], "smoke": "PASS"},
        "model_inference_tested": False, "weights_downloaded": False,
        "credentials_used": False, "video_generated": False,
    }


class IndependentKagglePreflightTests(unittest.TestCase):
    def test_valid_gpu_receipt_does_not_claim_model_readiness(self):
        report = validate_receipt(sample())
        self.assertEqual(report["status"], "CPU_AND_GPU_CHECK_PASSED")
        self.assertFalse(report["model_inference_verified"])
        self.assertFalse(report["accounts_or_api_access_verified"])
        self.assertFalse(report["pooled_vram"])

    def test_no_gpu_fails_closed(self):
        data = sample()
        data["gpu"] = {"torch_installed": False, "cuda_available": False,
                       "device_count": 0, "devices": [], "smoke": "SKIPPED"}
        report = validate_receipt(data)
        self.assertEqual(report["status"], "CPU_CHECK_PASSED_GPU_NOT_READY")

    def test_rejects_false_gpu_success_and_mismatched_count(self):
        data = sample()
        data["gpu"]["cuda_available"] = False
        with self.assertRaises(ValueError):
            validate_receipt(data)
        data = sample()
        data["gpu"]["device_count"] = 2
        with self.assertRaises(ValueError):
            validate_receipt(data)

    def test_rejects_claims_of_credential_access_or_video(self):
        for field in ("credentials_used", "weights_downloaded",
                      "model_inference_tested", "video_generated"):
            data = sample()
            data[field] = True
            with self.assertRaises(ValueError):
                validate_receipt(data)

    def test_cpu_hash_cannot_be_faked_with_wrong_value(self):
        data = sample()
        data["cpu"]["smoke_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_receipt(data)

    def test_notebook_is_self_contained_and_safe_to_parse(self):
        notebook = Path(__file__).resolve().parents[1] / "notebooks/TWO_WORLDS_KAGGLE_OWNER_CPU_GPU_CHECK.ipynb"
        data = json.loads(notebook.read_text(encoding="utf-8"))
        self.assertEqual(data["nbformat"], 4)
        scripts = ["".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(scripts), 1)
        code = scripts[0]
        for forbidden in ("KAGGLE_API_TOKEN", "requests.", "urllib.",
                          "kaggle kernels push", "subprocess.", "pip install", "git clone"):
            self.assertNotIn(forbidden, code)
        compile(code, str(notebook), "exec")

    def test_cpu_notebook_runs_offline_and_receipt_validates(self):
        notebook = Path(__file__).resolve().parents[1] / "notebooks/TWO_WORLDS_KAGGLE_OWNER_CPU_GPU_CHECK.ipynb"
        data = json.loads(notebook.read_text(encoding="utf-8"))
        code = "".join("".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                exec(compile(code, str(notebook), "exec"), {"__name__": "__main__"})
                receipt = json.loads(Path("two_worlds_compute_preflight.json").read_text(encoding="utf-8"))
                report = validate_receipt(receipt)
                self.assertIn(report["status"], ("CPU_CHECK_PASSED_GPU_NOT_READY",
                                                 "CPU_AND_GPU_CHECK_PASSED"))
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
