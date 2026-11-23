"""No GPU/network regression tests for immutable historical pilot readback."""
import hashlib
import unittest
from unittest.mock import patch

from tools.eb01_compare_existing_anatomy_pilots import (
    ORIGINAL_RUN_ID, original_kernel, compare_existing,
)
from tools.eb01_native_two_account_pilot import PilotBlocked


def motion(p95, drift=.23):
    return {
        "shot": "S06", "decoded_frames": 89, "review_flags": [],
        "regions": {
            "upper_skull": {
                "initial_tracks": 20, "expected_comparisons": 88,
                "measured_frames": 88, "p95_shape_residual_px": p95,
            },
            "background": {
                "initial_tracks": 20, "expected_comparisons": 88,
                "measured_frames": 88, "p95_displacement_px": drift,
            },
        },
    }


class PreviousPilotReadbackTests(unittest.TestCase):
    def test_immutable_kernel_slug_does_not_change(self):
        expected = hashlib.sha256(
            ("eb01-anatomy-v1-" + ORIGINAL_RUN_ID + "-owner").encode()
        ).hexdigest()[:14]
        self.assertEqual(original_kernel("owner", "private_alias"),
                         "private_alias/eb01-anatomy-" + expected)
        with self.assertRaises(ValueError):
            original_kernel("third", "private_alias")

    def test_read_only_commands_and_measurements_cannot_authorize_production(self):
        calls = []

        def fake_call(token, argv, **kwargs):
            calls.append(argv)
            if argv[1:3] == ["kernels", "status"]:
                return "KernelWorkerStatus.COMPLETE"
            if argv[1:3] == ["kernels", "output"]:
                return ""
            self.fail("Only existing Kaggle status and output may be called")

        with (
            patch("tools.eb01_compare_existing_anatomy_pilots.query",
                  return_value=("AUTHENTICATED_READ_ONLY", "private_alias")),
            patch("tools.eb01_compare_existing_anatomy_pilots.call",
                  side_effect=fake_call),
            patch("tools.eb01_compare_existing_anatomy_pilots.anatomy_specs",
                  return_value=[{"shot": "S06"}, {"shot": "S06"}]),
            patch("tools.eb01_compare_existing_anatomy_pilots.verify_slot",
                  side_effect=[
                      {"technical_pass": True, "regional_motion": motion(3.18)},
                      {"technical_pass": True, "regional_motion": motion(1.9)},
                  ]),
        ):
            result = compare_existing("owner", "private-token")
        self.assertEqual(len(calls), 2)
        self.assertEqual([c[1:3] for c in calls],
                         [["kernels", "status"], ["kernels", "output"]])
        self.assertEqual(result["original_mp4_count"], 2)
        self.assertTrue(result["paired_anatomy_metric"]["numerically_improved"])
        self.assertFalse(result["human_artistic_approved"])
        self.assertFalse(result["full_production_authorized"])
        self.assertEqual(result["new_kaggle_gpu_submissions"], 0)

    def test_missing_old_pilot_blocks_without_new_submission(self):
        calls = []

        def fake_call(token, argv, **kwargs):
            calls.append(argv)
            return "KernelWorkerStatus.ERROR"

        with (
            patch("tools.eb01_compare_existing_anatomy_pilots.query",
                  return_value=("AUTHENTICATED_READ_ONLY", "private_alias")),
            patch("tools.eb01_compare_existing_anatomy_pilots.call",
                  side_effect=fake_call),
        ):
            with self.assertRaises(PilotBlocked):
                compare_existing("owner", "private-token")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1:3], ["kernels", "status"])

    def test_failed_real_mp4_verification_never_becomes_comparison(self):
        with (
            patch("tools.eb01_compare_existing_anatomy_pilots.query",
                  return_value=("AUTHENTICATED_READ_ONLY", "private_alias")),
            patch("tools.eb01_compare_existing_anatomy_pilots.call",
                  return_value="KernelWorkerStatus.COMPLETE"),
            patch("tools.eb01_compare_existing_anatomy_pilots.anatomy_specs",
                  return_value=[{"shot": "S06"}, {"shot": "S06"}]),
            patch("tools.eb01_compare_existing_anatomy_pilots.verify_slot",
                  return_value={"technical_pass": False}),
        ):
            with self.assertRaises(PilotBlocked):
                compare_existing("owner", "private-token")
