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
