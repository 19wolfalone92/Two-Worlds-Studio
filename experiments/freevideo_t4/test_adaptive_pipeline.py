"""Offline regression tests. NO Kaggle, no ComfyUI, no model downloads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tw_t4_experiment", HERE / "adaptive_pipeline.py")
import sys
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class PolicyTests(unittest.TestCase):
    def test_t4_plan_excludes_unsupported_models_and_bf16(self):
        p = module.choose_plan(module.simulate_t4())
        self.assertEqual(p["status"], "HEURISTIC_PLAN_NOT_BENCHMARKED")
        self.assertEqual(p["backend"], "wan22-5b")
        self.assertFalse(p["bf16_allowed"])
        self.assertIn("freevideo_h3_not_t4_compatible", p["excluded"])
        self.assertEqual(p["final"], {"width": 576, "height": 1024, "frames": 49, "steps": 36})
        for stage in ("preview", "final"):
            v = p[stage]
            self.assertEqual(v["width"] * 16, v["height"] * 9)
            self.assertEqual((v["frames"] - 1) % 4, 0)

    def test_lower_free_vram_and_insufficient_hardware(self):
        lower = module.Hardware("T4 busy", "7.5", 16, 8, "detected_cuda")
        p = module.choose_plan(lower)
        self.assertEqual(p["profile"], "low_vram_cautious")
        self.assertEqual(p["final"]["width"], 432)
        with self.assertRaises(module.Blocked):
            module.choose_plan(module.Hardware("Old", "6.1", 8, 8, "detected_cuda"))
        with self.assertRaises(module.Blocked):
            module.choose_plan(module.Hardware("T4 overbooked", "7.5", 16, 4, "detected_cuda"))

    def test_hardware_simulation_can_only_plan(self):
        with patch.object(module, "detect_hardware", side_effect=AssertionError("unwanted GPU probe")):
            self.assertEqual(module.main(["plan", "--simulate-t4"]), 0)
        self.assertEqual(module.main(["preview", "--simulate-t4"]), 2)

    def test_no_execute_no_licence_must_fail_before_hardware(self):
        args = module.argparse.Namespace(command="preview", execute=False, accept_model_license=False)
        with patch.object(module, "detect_hardware", side_effect=AssertionError("unwanted GPU probe")):
            with self.assertRaises(module.Blocked):
                module.run(args)

    def test_loopback_security(self):
        module._check_host("http://127.0.0.1:8188")
        for uri in ("https://127.0.0.1:8188", "http://192.168.1.10:8188",
                    "http://localhost", "http://a:b@127.0.0.1:8188",
                    "http://127.0.0.1:8188/other"):
            with self.subTest(uri=uri), self.assertRaises(module.Blocked):
                module._check_host(uri)

    def test_explicit_review_bound_to_preview_and_reference_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref, clip, rev = root / "ref.png", root / "clip.mp4", root / "review.json"
            ref.write_bytes(b"reference bytes")
            clip.write_bytes(b"video bytes")
            good = {"schema": 1, "approved": True, "reviewer": "Owner",
                    "reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(),
                    "preview_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
                    "checks": dict.fromkeys(module.CHECKS, True)}
            rev.write_text(json.dumps(good))
            self.assertTrue(module.validate_review(rev, clip, ref)["approved"])
            clip.write_bytes(b"different video bytes")
            with self.assertRaisesRegex(module.Blocked, "SHA-256"):
                module.validate_review(rev, clip, ref)
            clip.write_bytes(b"video bytes")
            good["checks"]["anatomy"] = False
            rev.write_text(json.dumps(good))
            with self.assertRaisesRegex(module.Blocked, "checks"):
                module.validate_review(rev, clip, ref)

    def test_statistics_observed_only_not_invented_speedups(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.jsonl"
            for elapsed in (10.0, 12.0, 20.0):
                module._history_append(path, {"result": "GENERATED_REVIEW_REQUIRED",
                    "hardware": "Tesla T4", "stage": "preview",
                    "resolution": "432x768", "frames": 25, "elapsed_s": elapsed})
            output = module.summary(path)
            self.assertEqual(output["status"], "OBSERVED_BENCHMARKS_ONLY")
            self.assertEqual(output["groups"][0]["median_elapsed_s"], 12)
            self.assertEqual(output["groups"][0]["count"], 3)

    def test_private_output_guard(self):
        with self.assertRaisesRegex(module.Blocked, "OUTSIDE"):
            module._private_path(module.REPO_ROOT / "experiments" / "bad.mp4", must_exist=False)


if __name__ == "__main__":
    unittest.main()
