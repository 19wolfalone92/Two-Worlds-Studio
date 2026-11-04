"""Smooth camera continuity / subpixel video regression tests."""
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from studio.smooth_camera import camera_pose, render_still


class CameraStabilityTests(unittest.TestCase):
    def test_monotonic_eased_zoom_without_frame_jump(self):
        poses = [camera_pose(i, 90, "push_in") for i in range(90)]
        zooms = [p[0] for p in poses]
        velocities = [b-a for a,b in zip(zooms,zooms[1:])]
        self.assertTrue(all(v>=0 for v in velocities))
        self.assertLess(velocities[0], velocities[44]*.04)
        self.assertLess(velocities[-1], velocities[44]*.04)
        self.assertLess(max(velocities), .001)
        self.assertLess(zooms[-1]-zooms[0], .03)

    def test_camera_pan_and_invalid_strength(self):
        poses = [camera_pose(i, 90, "pan_left", .35) for i in range(90)]
        self.assertGreater(poses[0][1], poses[-1][1])
        self.assertAlmostEqual(poses[0][1], -poses[-1][1], places=8)
        self.assertTrue(all(math.isfinite(x) for p in poses for x in p))
        for invalid in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                camera_pose(0, 90, "push_in", invalid)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg missing")
    def test_real_30fps_subpixel_export(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)
            img=Image.new("RGB", (300,500), (30,60,90))
            d=ImageDraw.Draw(img)
            d.ellipse((50,100,220,310),fill=(230,160,90))
            src=path/"scene.png"
            img.save(src)
            out=path/"smooth.mp4"
            render_still(src, out, 360, 640, 30, 30, "push_in")
            self.assertGreater(out.stat().st_size,4000)
            info=subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
                                 "-show_entries","stream=nb_frames,r_frame_rate,width,height",
                                 "-of","default=noprint_wrappers=1",str(out)],
                                 check=True, capture_output=True,text=True).stdout
            self.assertIn("nb_frames=30",info)
            self.assertIn("r_frame_rate=30/1",info)
            self.assertIn("width=360",info)

if __name__ == "__main__":
    unittest.main()
