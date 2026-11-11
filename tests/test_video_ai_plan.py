"""Offline regression tests: current upstream quality plans and lost LK tracks."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from studio.motion_qc import _fit, _track
from studio.video_ai_plan import PROFILES, plan_wan_i2v


class VideoPlanTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.runtime = self.root / "official_Wan2.2"
        self.runtime.mkdir()
        (self.runtime / "generate.py").write_text("# official entry point stub for PLAN ONLY\n")
        self.weights = self.root / "staged_weights"
        self.weights.mkdir()
        self.reference = self.root / "start.png"
        Image.new("RGB", (576, 1024), (70, 100, 120)).save(self.reference)
        self.output = self.root / "private_output.mp4"
        self.prompt = (
            "A woman slowly turns her head toward the camera, her shoulders "
            "and breathing move naturally, hands remain anatomically correct, "
            "fixed indoor lighting and a stable lens."
        )

    def plan(self, profile="wan22_ti2v5b_quality", **kwargs):
        args = dict(runtime=self.runtime, weights=self.weights,
                    reference=self.reference, output=self.output,
                    prompt=self.prompt, seed=42)
        args.update(kwargs)
        return plan_wan_i2v(profile, **args)

    def test_five_b_uses_native_24fps_and_official_sampling(self):
        result = self.plan()
        argv = result["command_argv"]
        self.assertEqual(result["native_fps"], 24)
        self.assertEqual(result["frames"], 121)
        self.assertEqual(argv[argv.index("--task") + 1], "ti2v-5B")
        self.assertEqual(argv[argv.index("--sample_steps") + 1], "50")
        self.assertEqual(argv[argv.index("--sample_guide_scale") + 1], "5.0")
        self.assertEqual(argv[argv.index("--sample_shift") + 1], "5.0")
        self.assertEqual(argv[argv.index("--image") + 1], str(self.reference))
        self.assertFalse(result["inference_performed"])
        self.assertFalse(self.output.exists())

    def test_14b_keeps_dual_expert_guidance_in_upstream(self):
        result = self.plan("wan22_i2va14b_quality")
        argv = result["command_argv"]
        self.assertEqual(result["native_fps"], 16)
        self.assertEqual(result["frames"], 81)
        self.assertEqual(argv[argv.index("--task") + 1], "i2v-A14B")
        self.assertNotIn("--sample_guide_scale", argv)
        self.assertTrue(result["artistic_approval"])

    def test_t4_cannot_be_claimed_ready_for_official_720p(self):
        with self.assertRaisesRegex(ValueError, "free VRAM"):
            self.plan(free_vram_gib=15.9)
        self.assertFalse(self.output.exists())

    def test_wrong_frame_geometry_and_reuse_blocked(self):
        Image.new("RGB", (1024, 576)).save(self.reference)
        with self.assertRaisesRegex(ValueError, "9:16"):
            self.plan()
        Image.new("RGB", (576, 1024)).save(self.reference)
        self.output.write_bytes(b"already exists")
        with self.assertRaisesRegex(ValueError, "nonexisting"):
            self.plan()

    def test_no_unsupported_automatic_model_execution(self):
        self.assertEqual(PROFILES["ltx25_dfr_quality"]["status"],
                         "research_profile_no_unverified_runner")
        with self.assertRaisesRegex(ValueError, "verified only"):
            self.plan("wan_animate2_distilled")


class LostOpticalFlowTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((64, 64), dtype=np.uint8)
        self.boxes = [(0, 0, 64, 64)]

    def _fake(self, pass_number):
        class FakeCV:
            def __init__(self):
                self.calls = 0

            def goodFeaturesToTrack(self, *args, **kwargs):
                return np.array([[[16.0, 16.0]], [[32.0, 32.0]]], dtype=np.float32)

            def calcOpticalFlowPyrLK(self, before, after, points, unused, **kwargs):
                self.calls += 1
                if self.calls == pass_number:
                    return None, None, None
                return points.copy(), np.ones((len(points), 1), np.uint8), None

        return FakeCV()

    def test_forward_flow_disappears_without_crashing(self):
        cv = self._fake(1)
        pairs = _track(cv, self.frame, self.frame, self.boxes)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(pairs[0][0]), 0)
        self.assertEqual(cv.calls, 1)

    def test_reverse_flow_disappears_without_crashing(self):
        cv = self._fake(2)
        pairs = _track(cv, self.frame, self.frame, self.boxes)
        self.assertEqual(len(pairs[0][0]), 0)
        self.assertEqual(cv.calls, 2)

    def test_ransac_missing_inliers_does_not_crash(self):
        class FakeCV:
            RANSAC = 8
            def estimateAffinePartial2D(self, *args, **kwargs):
                return np.eye(2, 3), None

        before = np.array([[i * 10, i * 20 % 41] for i in range(8)], dtype=np.float32)
        after = before.copy()
        self.assertIsNone(_fit(FakeCV(), (before, after)))


if __name__ == "__main__":
    unittest.main()
