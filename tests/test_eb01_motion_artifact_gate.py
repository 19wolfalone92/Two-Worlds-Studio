"""CPU-only continuity QA: reject artifacted frames before claiming a motion pilot."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from studio.eb01_motion_artifact_gate import (
    ArtifactRejected, bright_color_fraction, classify,
    inspect_clip, trim_bad_tail, neon_streak_line_geometry,
    narrow_neon_intrusion_fraction
)


class E01MotionArtifactTests(unittest.TestCase):
    def test_stripe_statistics_on_dark_kostya_frame(self):
        f=np.full((288,160,3),20,dtype=np.uint8)
        self.assertLess(bright_color_fraction(f),.001)
        f[265:,:,0]=254
        f[265:,:,1]=200
        f[265:,:,2]=8
        self.assertGreater(bright_color_fraction(f),.10)

    def test_actual_incident_three_terminal_corrupt_frames(self):
        baseline=[.0001]*46
        frac=baseline+[.0971,.1008,.0639]
        result=classify(frac)
        self.assertEqual(result["status"],"TRIM_ONLY_TERMINAL_ARTIFACTS")
        self.assertEqual(result["safe_frame_count"],46)
        self.assertEqual(result["flagged_frames"],[46,47,48])

    def test_random_interior_glitch_forces_full_rejection(self):
        v=[.0001]*24
        v[9]=.15
        self.assertEqual(classify(v)["status"],"REJECT_NONTERMINAL_NEON_ARTIFACT")
        self.assertEqual(classify(v)["safe_frame_count"],0)

    def test_no_artifact_does_not_fabricate_motion_approval(self):
        v=[.001]*49
        self.assertEqual(classify(v)["status"],"NO_NEON_ARTIFACT_FOUND")
        self.assertEqual(classify(v)["safe_frame_count"],49)

    def test_blue_palette_is_not_automatically_thin_neon_stripe(self):
        frame=np.full((128,160,3),[22,21,20],dtype=np.uint8)
        frame[20:110,12:148,:]=[10,35,215]
        self.assertGreater(narrow_neon_intrusion_fraction(frame),.1)
        self.assertEqual(neon_streak_line_geometry(frame),0.0)

    def test_true_two_row_rgb_streak_produces_independent_line_evidence(self):
        frame=np.full((128,160,3),[22,21,20],dtype=np.uint8)
        frame[55:57,12:148,:]=[10,35,215]
        self.assertGreater(neon_streak_line_geometry(frame),.005)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg/ffprobe not installed")
    def test_persistent_blue_palette_review_only_never_false_green(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for shape,expected in (
                ("wide","REVIEW_PERSISTENT_COOL_CHROMA_NO_STRUCTURAL_PROOF"),
                ("line","REJECT_STRUCTURAL_RGB_STREAK_LINES"),
            ):
                frames=[]
                for i in range(21):
                    frame=np.full((128,160,3),[22,21,20],dtype=np.uint8)
                    if shape=="wide":
                        frame[20:110,12:148,:]=[10,35,215]
                    else:
                        frame[55:57,12:148,:]=[10,35,215]
                    frames.append(frame.tobytes())
                video=root/(shape+".mp4")
                cmd=["ffmpeg","-hide_banner","-nostdin","-y","-loglevel","error",
                     "-f","rawvideo","-pix_fmt","rgb24","-s","160x128","-r","24",
                     "-i","-","-an","-c:v","libx264","-crf","10",
                     "-pix_fmt","yuv420p",str(video)]
                subprocess.run(cmd,input=b"".join(frames),check=True,
                               capture_output=True,timeout=45)
                report=inspect_clip(video)
                self.assertEqual(report["status"],expected,report)
                self.assertEqual(report["safe_frame_count"],0)
                self.assertFalse(report["actor_motion_artistically_approved"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg/ffprobe not installed")
    def test_real_encoded_neon_tail_is_detected_and_only_tail_trimmed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            src=root/"fake_motion.mp4"
            out=root/"cleaned_review.mp4"
            frames=[]
            for i in range(21):
                frame=np.zeros((288,160,3),dtype=np.uint8)
                frame[:]=[26+i,23+i,20+i]
                # Use bright vivid green and blue in the bottom 14% to emulate
                # the ACTUAL final two neon bars (not an approved art image).
                if i>=19:
                    frame[int(288*.86):,:,:]=[15,238,235]
                frames.append(frame.tobytes())
            cmd=["ffmpeg","-hide_banner","-nostdin","-y","-loglevel","error",
                 "-f","rawvideo","-pix_fmt","rgb24","-s","160x288","-r","24",
                 "-i","-","-an","-c:v","libx264","-crf","16","-pix_fmt","yuv420p",str(src)]
            subprocess.run(cmd,input=b"".join(frames),check=True,capture_output=True,timeout=45)
            result=inspect_clip(src)
            self.assertEqual(result["status"],"TRIM_ONLY_TERMINAL_ARTIFACTS")
            self.assertEqual(result["safe_frame_count"],19)
            report=trim_bad_tail(src,out)
            self.assertEqual(report["source_frames"],21)
            self.assertEqual(report["result_frames"],19)
            self.assertFalse(report["human_motion_approval"])
            self.assertFalse(report["published"])
            self.assertEqual(inspect_clip(out)["status"],"NO_NEON_ARTIFACT_FOUND")


if __name__=="__main__":
    unittest.main()
