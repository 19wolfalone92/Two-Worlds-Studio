"""Private real-art mirror renderer — deterministic synthetic-only regressions."""
from __future__ import annotations
import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:
    cv2=None


@unittest.skipUnless(cv2 is not None,'OpenCV installed only on private/CI animation worker')
class RealArtSceneContractTests(unittest.TestCase):
    def setUp(self):
        from studio.private_real_art_scene import MirrorMotionScene
        self.MirrorMotionScene=MirrorMotionScene
        w,h=188,334
        self.im=Image.new('RGB',(w,h),(14,22,38))
        d=ImageDraw.Draw(self.im)
        d.ellipse((92,10,176,236),fill=(44,64,96))
        d.rectangle((18,75,85,316),fill=(72,87,114))
        d.ellipse((23,35,73,100),fill=(50,37,30))
        d.ellipse((119,61,160,113),fill=(169,117,99))

    def test_motion_changes_real_pixels_but_stays_deterministic(self):
        scene=self.MirrorMotionScene(self.im,160,284)
        start=scene.frame(0)
        end=scene.frame(8)
        self.assertEqual(start.shape,(284,160,3))
        self.assertGreater(float(np.abs(start.astype('int16')-end.astype('int16')).mean()),.05)
        self.assertEqual(scene.frame(8).tobytes(),end.tobytes())

    def test_reject_wrong_shape_and_bad_clock(self):
        with self.assertRaisesRegex(ValueError,'vertical'):
            self.MirrorMotionScene(Image.new('RGB',(100,100)),160,284)
        scene=self.MirrorMotionScene(self.im,160,284)
        for sec in (-1,10.5,float('nan')):
            with self.assertRaises(ValueError):
                scene.frame(sec)

    def test_missing_real_art_does_not_generate_fake_actors(self):
        from studio.private_real_art_scene import render_private
        with tempfile.TemporaryDirectory() as d:
            dst=Path(d)/'private.mp4'
            with self.assertRaises(FileNotFoundError):
                render_private(Path(d)/'missing_owner_art.png',dst)
            self.assertFalse(dst.exists())

    def test_review_gate_blocks_clobbering(self):
        from studio.private_real_art_scene import render_private
        with tempfile.TemporaryDirectory() as d:
            dst=Path(d)/'existing.mp4';dst.write_bytes(b'preserve')
            with self.assertRaisesRegex(FileExistsError,'overwrite'):
                render_private(Path(d)/'missing.png',dst)
            self.assertEqual(dst.read_bytes(),b'preserve')
