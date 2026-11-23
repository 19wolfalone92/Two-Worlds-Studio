import importlib.util
import unittest

import numpy as np
from studio.eb01_regional_motion import _track_regions, rigid_residual


class RegionalMotionTests(unittest.TestCase):
    def test_rigid_rotation_and_translation_are_not_shape_changes(self):
        a = np.array([[1., 0.], [3., 1.], [2., 5.], [5., 3.]])
        angle = .17
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        b = a @ rotation + [11., -5.]
        self.assertLess(rigid_residual(a, b).max(), 1.e-6)
        b[2] += [8, 0]
        self.assertGreater(rigid_residual(a, b).max(), 2.5)

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'cinematic dependencies required')
    def test_flat_region_is_unknown_and_textured_translation_keeps_its_shape(self):
        import cv2
        rng = np.random.default_rng(42)
        a = np.zeros((96, 160), np.uint8)
        a[20:70, 15:75] = rng.integers(0, 255, (50, 60), dtype=np.uint8)
        b = cv2.warpAffine(a, np.float32([[1, 0, 4], [0, 1, 1]]), (160, 96))
        count, report, flags = _track_regions(a, [b], {'hand': (15, 20, 75, 70), 'blank': (90, 10, 150, 80)})
        self.assertEqual(count, 2)
        self.assertIn('blank:insufficient_tracking', flags)
        self.assertLess(report['hand']['p95_shape_residual_px'], .2)
        self.assertAlmostEqual(report['hand']['p95_displacement_px'], np.sqrt(17), delta=.2)

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'cinematic dependencies required')
    def test_missing_flow_result_is_unknown_not_a_crash_or_approval(self):
        import cv2
        from unittest.mock import patch
        rng = np.random.default_rng(71)
        initial = rng.integers(0, 256, (96, 160), dtype=np.uint8)
        with patch.object(cv2, 'calcOpticalFlowPyrLK', return_value=(None, None, None)) as flow:
            frames, report, flags = _track_regions(
                initial, [initial.copy(), initial.copy()], {'hand': (10, 10, 140, 85)}
            )
        self.assertEqual(frames, 3)
        self.assertEqual(flow.call_count, 2)  # No invalid backward call.
        self.assertEqual(report['hand']['measured_frames'], 0)
        self.assertEqual(report['hand']['expected_comparisons'], 2)
        self.assertIn('hand:insufficient_tracking', flags)

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'cinematic dependencies required')
    def test_nan_failed_forward_tracks_never_reach_backward_solver(self):
        import cv2
        from unittest.mock import patch
        rng = np.random.default_rng(72)
        initial = rng.integers(0, 256, (96, 160), dtype=np.uint8)

        def failed_forward(old_frame, new_frame, points, result, **kwargs):
            self.assertIsNone(result)
            return np.full_like(points, np.nan), np.zeros((len(points), 1), np.uint8), None

        with patch.object(cv2, 'calcOpticalFlowPyrLK', side_effect=failed_forward) as flow:
            frames, report, flags = _track_regions(
                initial, [np.zeros_like(initial)], {'hand': (10, 10, 140, 85)}
            )
        self.assertEqual(frames, 2)
        self.assertEqual(flow.call_count, 1)
        self.assertEqual(report['hand']['measured_frames'], 0)
        self.assertIn('hand:insufficient_tracking', flags)

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'cinematic dependencies required')
    def test_invalid_backward_tracks_are_never_counted_as_proof(self):
        import cv2
        from unittest.mock import patch
        rng = np.random.default_rng(73)
        initial = rng.integers(0, 256, (96, 160), dtype=np.uint8)
        seen = [0]

        def bad_inverse(old_frame, new_frame, points, result, **kwargs):
            seen[0] += 1
            if seen[0] == 1:
                return points.copy(), np.ones((len(points), 1), np.uint8), None
            return np.full_like(points, np.nan), np.ones((len(points), 1), np.uint8), None

        with patch.object(cv2, 'calcOpticalFlowPyrLK', side_effect=bad_inverse):
            frames, report, flags = _track_regions(
                initial, [initial.copy()], {'hand': (10, 10, 140, 85)}
            )
        self.assertEqual(seen[0], 2)
        self.assertEqual(frames, 2)
        self.assertEqual(report['hand']['measured_frames'], 0)
        self.assertIn('hand:insufficient_tracking', flags)
