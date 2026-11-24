"""Native Studio-wave invariants: no fake readiness and no forgotten imports."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.studio_wave_status_gate import (
    JobStateError, PENDING, SOURCE_SHA256, classify, parse_state, verify_actual_native_mp4,
)

ROOT=Path(__file__).resolve().parents[1]

class NativeWaveGateTests(unittest.TestCase):
    def test_kaggle_cli_is_parsed_without_inventing_a_state(self):
        self.assertEqual(parse_state('my-kernel has status "KernelWorkerStatus.QUEUED"'),"QUEUED")
        self.assertEqual(parse_state('KernelWorkerStatus.RUNNING'),"RUNNING")
        self.assertEqual(parse_state('KernelWorkerStatus.COMPLETE'),"COMPLETE")
        for bad in ("", "READY", "KernelWorkerStatus.BORED",
                    "KernelWorkerStatus.COMPLETE\\nKernelWorkerStatus.ERROR"):
            with self.subTest(bad=bad),self.assertRaises(JobStateError):
                parse_state(bad)

    def test_queue_is_not_error_and_must_not_duplicate_gpu_job(self):
        for state in PENDING:
            data=classify(state)
            self.assertFalse(data["model_failure_confirmed"])
            self.assertFalse(data["real_mp4_accepted"])
            self.assertFalse(data["safe_to_requeue"])
            self.assertTrue(data["requires_later_readback"])
        self.assertEqual(classify("QUEUED")["status"],"WAITING_GPU")
        self.assertEqual(classify("RUNNING")["status"],"GPU_RUNNING")

    def test_terminal_kaggle_state_cannot_be_called_proven_neural_video(self):
        self.assertFalse(classify("COMPLETE")["real_mp4_accepted"])
        for state in ("ERROR","CANCEL","CANCELLED","CANCEL_ACKNOWLEDGED"):
            with self.subTest(state=state):
                self.assertEqual(classify(state)["status"],"KAGGLE_TERMINAL_FAILURE")
                self.assertFalse(classify(state)["safe_to_requeue"])
        with self.assertRaises(JobStateError):
            classify("FINISHED")

    def test_missing_or_forged_video_proof_is_rejected_before_ffprobe(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(JobStateError,"missing"):
                verify_actual_native_mp4(root)
            (root/"STUDIO_WAVE_TRUE_NATIVE_65F_24FPS.mp4").write_bytes(b"not-video"*2000)
            (root/"STUDIO_WAVE_TRUE_NATIVE_RECEIPT.json").write_text(json.dumps({
                "schema":1,"status":"NATIVE_NEURAL_VIDEO_GENERATED_ARTISTIC_REVIEW_REQUIRED",
                "model":"Lightricks/LTX-Video-2B-0.9.6-distilled",
                "source_reference_sha256":SOURCE_SHA256,
                "source":"original_user_reference_jpeg_only_not_previous_video_frames",
                "actual_native_gpu_inference":False,
                "output_frames":65,"output_width":576,"output_height":1024,"output_fps":24,
                "extra_interpolated_frames":0,"private_media_only":True,"published":False,
                "human_artistic_approved":False,
                "anatomical_continuity_human_review_required":True,
            }))
            with self.assertRaisesRegex(JobStateError,"receipt policy mismatch"):
                verify_actual_native_mp4(root)

    def test_actual_native_entry_has_no_missing_python_import(self):
        file=ROOT/"tools/kaggle_studio_man_wave_native.py"
        source=file.read_text(encoding="utf-8")
        tree=ast.parse(source)
        mods=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module and (
                    node.module.startswith("studio.") or node.module.startswith("tools.")):
                mods.add(node.module)
        self.assertGreaterEqual(len(mods),4)
        for module in sorted(mods):
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.util.find_spec(module),
                                     "Studio GPU import missing: "+module)
        self.assertEqual(source.count('__STUDIO_WAVE_IMAGE_B64__'),1)
        self.assertEqual(source.count('__STUDIO_WAVE_CONFIG_B64__'),1)

    def test_forensic_cannot_launch_gpu_or_replace_real_model(self):
        source=(ROOT/"tools/studio_wave_private_forensic.py").read_text(encoding="utf-8")
        tree=ast.parse(source)
        self.assertNotIn('kernels","push',source)
        self.assertNotIn('kernel fake',source.lower())
        self.assertIsInstance(tree,ast.AST)

if __name__=="__main__":
    unittest.main()
