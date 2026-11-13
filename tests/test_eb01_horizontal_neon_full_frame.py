"""CPU regression for full-height neon stripe detection; no private assets."""
import unittest
import numpy as np
from studio.eb01_motion_artifact_gate import horizontal_neon_band_fraction, bright_color_fraction


class NeonAcrossFrameTests(unittest.TestCase):
    def test_cyan_band_above_bottom_fourteen_percent(self):
        frame = np.full((576, 320, 3), 42, dtype=np.uint8)
        frame[150:164, 20:300] = (12, 240, 245)
        self.assertEqual(bright_color_fraction(frame), 0.0)
        self.assertGreater(horizontal_neon_band_fraction(frame), .012)

    def test_red_band_above_bottom_fourteen_percent(self):
        frame = np.full((576, 320, 3), 42, dtype=np.uint8)
        frame[200:215, 10:310] = (250, 10, 30)
        self.assertGreater(horizontal_neon_band_fraction(frame), .012)

    def test_small_colored_prop_not_broad_stripe(self):
        frame = np.full((576, 320, 3), 42, dtype=np.uint8)
        frame[100:130, 140:180] = (250, 10, 30)
        self.assertEqual(horizontal_neon_band_fraction(frame), 0.0)


if __name__ == "__main__":
    unittest.main()
