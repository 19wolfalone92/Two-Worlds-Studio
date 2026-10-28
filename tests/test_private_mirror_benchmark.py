"""Safe unit tests: no private Work artwork is included in GitHub."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
try:
    import cv2
except ImportError:
    cv2=None

@unittest.skipUnless(cv2 is not None, 'Optional opencv-python-headless not installed')
class WorkMirrorBenchmarkTests(unittest.TestCase):
    def setUp(self):
        from studio import private_mirror_benchmark as mod
        self.mod=mod
    def test_arm_split_never_adds_painted_pixels(self):
        src=np.zeros((192,108,4),np.uint8)
        src[43:164,14:79]=[90,120,155,255]
        body,arm,pivot=self.mod.prep_arm(src,'mira')
        self.assertEqual(src.shape,body.shape)
        self.assertTrue(np.all(body[:,:,3]<=src[:,:,3]))
        self.assertTrue(np.all(arm[:,:,3]<=src[:,:,3]))
        self.assertGreater(int(arm[:,:,3].max()),0)
        self.assertGreater(int(body[:,:,3].max()),0)
    def test_missing_private_work_zip_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                self.mod.art(Path(d)/'absent.zip')
    def test_continuity_of_keyframe_easing(self):
        e=self.mod.eased
        self.assertEqual(e(0),0)
        self.assertEqual(e(1),1)
        self.assertAlmostEqual(e(.25),.15625)
        self.assertTrue(e(.25)<e(.50)<e(.75))
