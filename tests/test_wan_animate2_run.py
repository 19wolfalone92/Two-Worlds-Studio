"""Offline Wan-Animate-2 native runner and locked driver import contract tests."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from studio.comfy_neural import _probe
from studio.hybrid_master import check_manifest
from studio.release_gate import sha256
from studio.wan_animate2_run import MODEL, import_take, plan, generate, NeuralBlocked


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class WanAnimate2Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.runtime = self.root / "Wan-Animate-2"
        infer = self.runtime / "infer"
        infer.mkdir(parents=True)
        (infer / "wan_animate_2_demo.py").write_text("# no GPU called\n")
        (infer / "wan_animate_2_distillation.yaml").write_text("test_only: true\n")
        self.reference = self.root / "reference.png"
        Image.new("RGB", (576, 1024), (80, 50, 30)).save(self.reference)
        self.driver = self.root / "driver.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi",
            "-i", "testsrc2=s=576x1024:r=24", "-frames:v", "17",
            "-c:v", "libx264", "-preset", "ultrafast", str(self.driver)
        ], check=True)
        self.clip = self.root / "animated.mp4"
        shutil.copy2(self.driver, self.clip)
        self.prompt = "A woman stands near the doorway with detailed hair and leather coat."

    def test_official_distilled_gpu_invocation_is_plan_not_synthetic_motion(self):
        request = plan(runtime=self.runtime, reference=self.reference, driver=self.driver,
                       output=self.root / "fresh.mp4", prompt=self.prompt, frames=17)
        self.assertEqual(request["model"], MODEL)
        self.assertEqual(request["steps"], 10)
        self.assertEqual(request["guidance"], 1.)
        self.assertEqual(request["fps"], 24)
        self.assertEqual(request["width"], 480)
        self.assertEqual(request["height"], 848)
        self.assertFalse(request["gpu_executed"])
        with self.assertRaises(NeuralBlocked):
            generate(runtime=self.runtime, reference=self.reference, driver=self.driver,
                     output=self.root / "fresh.mp4", prompt=self.prompt,
                     accepted_license=False, frames=17)

    def _receipt(self):
        output = self.clip.with_suffix(".receipt.json")
        payload = {
            "schema": 1, "engine": "neural_animate2", "model": MODEL,
            "source_revision": "a" * 40, "config_sha256": "b" * 64,
            "reference_sha256": sha256(self.reference),
            "driver_sha256": sha256(self.driver),
            "clip_sha256": sha256(self.clip),
            "video": _probe(self.clip), "inference_performed": True,
        }
        output.write_text(json.dumps(payload))
        return output

    def test_wan_animate2_clip_integrates_and_its_driver_is_locked(self):
        receipt = self._receipt()
        imported = import_take(self.clip, receipt, self.reference, self.driver, duration=17/24)
        self.assertEqual(imported["model"], MODEL)
        manifest = {
            "schema": 1, "visual_profile": "photoreal_cinematic",
            "shots": [{
                "clip": self.clip.name, "duration": 17/24,
                "clip_take": {
                    "engine": "neural_animate2", "visual_profile": "photoreal_cinematic",
                    "sha256": sha256(self.clip), "receipt": receipt.name,
                    "reference": self.reference.name, "driver": self.driver.name,
                }
            }]
        }
        normalized = check_manifest(manifest, self.root)
        self.assertEqual(normalized["shots"][0]["animation_engine"], "neural_animate2")
        manifest.update({"editorial_approved": True, "rights_approved": True})
        manifest["shots"][0]["approved"] = True
        with self.assertRaisesRegex(ValueError, "independent character-motion QC"):
            check_manifest(manifest, self.root, master=True)
        Image.new("RGB", (576, 1024), (240, 10, 10)).save(self.reference)
        with self.assertRaisesRegex(ValueError, "source and full clip"):
            import_take(self.clip, receipt, self.reference, self.driver, duration=17/24)

    def test_bad_driver_duration_blocks_too_long_request(self):
        with self.assertRaisesRegex(ValueError, "shorter"):
            plan(runtime=self.runtime, reference=self.reference, driver=self.driver,
                 output=self.root / "fresh.mp4", prompt=self.prompt, frames=81)


if __name__ == "__main__":
    unittest.main()
