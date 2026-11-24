"""REAL synthetic FFmpeg fixtures; no GPU or generative model is involved."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("freevideo_cpu_pixel_metrics", HERE / "quality_metrics.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class QualityMetricsTests(unittest.TestCase):
    def make_movie(self, target, source):
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", source,
                        "-frames:v", "25", "-c:v", "mpeg4", "-q:v", "8", str(target)],
                       capture_output=True, check=True, timeout=30)

    def test_static_video_is_flagged_as_frozen_not_neural(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "static.mp4"
            self.make_movie(path, "color=c=black:s=192x320:r=24")
            report = module.audit_video(path)
            self.assertEqual(report["frames"], 25)
            self.assertEqual(report["status"], "CPU_PIXEL_METRICS_ONLY")
            self.assertEqual(report["artistic_quality"], "NOT_VERIFIED")
            self.assertEqual(report["neural_inference"], "NOT_VERIFIED")
            self.assertEqual(report["near_duplicate_pairs"], 24)
            self.assertIs(type(report["near_duplicate_pairs"]), int)
            self.assertIs(type(report["abrupt_pixel_change_pairs"]), int)
            self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
            self.assertIn("POSSIBLE_FROZEN_OR_REPEATED_FRAMES", report["flags"])

    def test_pixel_change_does_not_prove_character_animation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "changes.mp4"
            self.make_movie(path, "testsrc=s=192x320:r=24")
            report = module.audit_video(path)
            self.assertGreater(report["adjacent_mean_absolute_difference_p95"], 0)
            self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
            self.assertEqual(report["artistic_quality"], "NOT_VERIFIED")

    def test_corrupt_video_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.mp4"
            path.write_bytes(b"0" * 4096)
            with self.assertRaises(module.VideoRejected):
                module.audit_video(path)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "real.mp4"
            self.make_movie(target, "color=c=black:s=192x320:r=24")
            alias = Path(tmp) / "alias.mp4"
            alias.symlink_to(target)
            with self.assertRaises(module.VideoRejected):
                module.audit_video(alias)

if __name__ == "__main__":
    unittest.main()
