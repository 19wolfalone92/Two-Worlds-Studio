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

class RearViewStagingContractTests(unittest.TestCase):
    """No private art needed: reject the wrong character orientation and stiffer rigs."""

    def test_rear_view_requires_seven_animated_body_parts(self):
        from studio.mirror_rear_rig import validate_mira_rear_contract
        names=("torso","head","upper_arm","forearm","hand","left_leg","right_leg")
        tracks={name:[{"at":0,"angle":0},{"at":1,"angle":4}]
                for name in ("head","upper_arm","forearm","left_leg","right_leg")}
        tracks["__root__"]=[{"at":0,"x":-.04},{"at":1,"x":.03}]
        cfg={"schema":1,"body_id":"earth_mira","camera_view":"back",
             "facing":"mirror",
             "parts":[{"name":name,"sha256":"a"*64} for name in names],
             "tracks":tracks,
             "ground_shadow":{"center":[.5,.90],"radii":[.15,.025]}}
        validate_mira_rear_contract(cfg)
        for wrong in ("front","three_quarter","side",None):
            bad=dict(cfg,camera_view=wrong)
            with self.assertRaisesRegex(ValueError,"BACK"):
                validate_mira_rear_contract(bad)
        bad=dict(cfg,facing="camera")
        with self.assertRaisesRegex(ValueError,"BACK"):
            validate_mira_rear_contract(bad)
        bad=dict(cfg,parts=cfg["parts"][:-1])
        with self.assertRaisesRegex(ValueError,"right_leg"):
            validate_mira_rear_contract(bad)
        bad=dict(cfg,tracks={"__root__":tracks["__root__"]})
        with self.assertRaisesRegex(ValueError,"articulated motion"):
            validate_mira_rear_contract(bad)
        bad=dict(cfg,tracks=dict(tracks,head=[{"at":0},{"at":1}]))
        with self.assertRaisesRegex(ValueError,"head"):
            validate_mira_rear_contract(bad)

    def test_render_fails_closed_without_private_rear_art(self):
        from studio.private_mirror_parallel import run
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,"genuine private"):
                run(Path(temp)/"work.zip",Path(temp)/"preview.mp4")
            self.assertFalse((Path(temp)/"preview.mp4").exists())
