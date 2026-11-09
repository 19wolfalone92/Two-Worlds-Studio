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

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg missing')
    def test_static_hold_keeps_the_same_composition_and_exact_frame_count(self):
        import json
        from PIL import ImageOps
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source.png';out=root/'hold.mp4'
            im=Image.new('RGB',(310,510),(30,60,90));ImageDraw.Draw(im).rectangle((72,122,200,360),fill=(220,160,80));im.save(source)
            render_still(source,out,360,640,30,42,'static')
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(out)]))['streams'][0]
            self.assertEqual(info['nb_frames'],'42')
            frames=subprocess.check_output(['ffmpeg','-v','error','-i',str(out),'-f','framemd5','-']).decode()
            hashes=[x.split(',')[-1].strip() for x in frames.splitlines() if x and not x.startswith('#')]
            self.assertEqual(len(set(hashes)),1)
            # The static optimization must preserve the old overscan crop.
            raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(out),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'])
            actual=Image.frombytes('RGB',(360,640),raw)
            iw,ih=math.ceil(360*1.12),math.ceil(640*1.12)
            fitted=ImageOps.fit(im,(iw,ih),method=Image.Resampling.LANCZOS)
            expected=fitted.transform((360,640),Image.Transform.AFFINE,(1,0,(iw-360)/2,0,1,(ih-640)/2),resample=Image.Resampling.BICUBIC)
            import numpy as np
            self.assertLess(float(np.abs(np.asarray(actual,dtype=float)-np.asarray(expected,dtype=float)).mean()),3)

if __name__ == "__main__":
    unittest.main()
