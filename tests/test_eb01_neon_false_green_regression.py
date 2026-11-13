"""Block color-corrupted EB01 clips even when stripes exist from frame zero.

This is a CPU unit test, not a claim that a decoded video is visually good.
"""
import unittest
import numpy as np

from studio.eb01_motion_artifact_gate import narrow_neon_intrusion_fraction


class EB01NeonFalseGreenRegression(unittest.TestCase):
    def test_thin_magenta_glitch_is_detected_in_warm_scene(self):
        frame = np.full((576, 320, 3), [55, 42, 28], dtype=np.uint8)
        self.assertEqual(narrow_neon_intrusion_fraction(frame), 0.0)
        frame[164:166] = [173, 49, 225]
        self.assertGreater(narrow_neon_intrusion_fraction(frame), .0015)

    def test_genuine_warm_gothic_color_does_not_trigger(self):
        frame = np.full((576, 320, 3), [148, 94, 51], dtype=np.uint8)
        self.assertEqual(narrow_neon_intrusion_fraction(frame), 0.0)

    def test_single_pixel_or_artifact_does_not_force_color_judgment(self):
        frame = np.full((576, 320, 3), [66, 39, 29], dtype=np.uint8)
        frame[170, 260] = [173, 49, 225]
        self.assertLess(narrow_neon_intrusion_fraction(frame), .0015)


if __name__ == "__main__":
    unittest.main()
