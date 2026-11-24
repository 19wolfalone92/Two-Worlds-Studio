"""Offline source-stage tests: no actual GPU or network calls."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
from tools.eb01_bounded_inference import (
    run_native_bounded,InferenceTimeBudgetExpired,latest_stage,
)

class SafeNativeStageMonitorTests(unittest.TestCase):
    def test_real_completed_subprocess_must_print_approved_native_output_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            result=run_native_bounded(
                [sys.executable,"-u","-c",
                 "print('EB01_STAGE_BEGIN_MODEL_LOAD',flush=True);"
                 "print('EB01_STAGE_DENOISE_STEP_END 8',flush=True);"
                 "print('EB01_STAGE_NATIVE_VIDEO_OUTPUT_READY',flush=True)"],
                cwd=root,env={},log_path=root/"stream.txt",
                deadline_seconds=10,heartbeat_seconds=1)
            self.assertTrue(result["native_pipeline_completed"])
            self.assertEqual(result["last_stage"],"NATIVE_VIDEO_OUTPUT_READY")
    def test_exit_zero_without_real_output_stage_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(RuntimeError,"without proved native video"):
                run_native_bounded(
                    [sys.executable,"-u","-c","print('EB01_STAGE_BEGIN_MODEL_LOAD')"],
                    cwd=root,env={},log_path=root/"log",
                    deadline_seconds=10,heartbeat_seconds=1)
    def test_strict_marker_never_infers_success_from_user_supplied_text(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"log"
            path.write_text("secret prompt: EB01_STAGE_NATIVE_VIDEO_OUTPUT_READY\n"
                            "EB01_STAGE_WRONG_BAD_STATE\n"
                            "EB01_STAGE_DENOISE_STEP_END 2\n",encoding="utf-8")
            self.assertEqual(latest_stage(path),"DENOISE_STEP_END")
    def test_bounded_live_child_timeout_kills_group(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            now=[0.0]
            def clock():
                now[0]+=11
                return now[0]
            with self.assertRaises(InferenceTimeBudgetExpired):
                run_native_bounded(
                    [sys.executable,"-u","-c",
                     "import time;print('EB01_STAGE_BEGIN_MODEL_LOAD',flush=True);time.sleep(60)"],
                    cwd=root,env={},log_path=root/"log",deadline_seconds=10,
                    heartbeat_seconds=1,monotonic=clock)
    def test_wave_native_marker_is_verified_without_being_confused_with_eb01(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            log=root/"gpu.log"
            child=("print('WAVE_STAGE_BEGIN_MODEL_LOAD',flush=True);"
                   "print('WAVE_STAGE_BEGIN_TRANSFORMER',flush=True);"
                   "print('WAVE_STAGE_END_DECODE',flush=True);"
                   "print('WAVE_STAGE_NATIVE_VIDEO_OUTPUT_READY',flush=True)")
            result=run_native_bounded(
                [sys.executable,"-u","-c",child],
                cwd=root,env={},log_path=log,
                deadline_seconds=10,heartbeat_seconds=1,stage_prefix="WAVE")
            self.assertEqual(result["last_stage"],"NATIVE_VIDEO_OUTPUT_READY")
            self.assertTrue(result["native_pipeline_completed"])
            self.assertEqual(latest_stage(log), "NO_NATIVE_STAGE_MARKER")
            self.assertEqual(latest_stage(log,stage_prefix="WAVE"), "NATIVE_VIDEO_OUTPUT_READY")

    def test_cross_project_marker_must_not_accept_false_native_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            child="print('EB01_STAGE_NATIVE_VIDEO_OUTPUT_READY',flush=True)"
            with self.assertRaisesRegex(RuntimeError,"NO_NATIVE_STAGE_MARKER"):
                run_native_bounded(
                    [sys.executable,"-u","-c",child],
                    cwd=root,env={},log_path=root/"gpu.log",
                    deadline_seconds=10,heartbeat_seconds=1,stage_prefix="WAVE")

    def test_invalid_stage_prefix_rejected_without_running_child(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(ValueError,"Unsupported"):
                run_native_bounded(["does-not-exist"],cwd=root,env={},
                                   log_path=root/"gpu.log",deadline_seconds=10,
                                   heartbeat_seconds=1,stage_prefix="WAVE\\nEB01")
            self.assertFalse((root/"gpu.log").exists())

    def test_subprocess_zero_requires_final_stage_not_just_decode(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            child=("print('WAVE_STAGE_BEGIN_DECODE',flush=True);"
                   "print('WAVE_STAGE_END_DECODE',flush=True)")
            with self.assertRaisesRegex(RuntimeError,"last_stage=END_DECODE"):
                run_native_bounded(
                    [sys.executable,"-u","-c",child],
                    cwd=root,env={},log_path=root/"gpu.log",
                    deadline_seconds=10,heartbeat_seconds=1,stage_prefix="WAVE")

    def test_invalid_unbounded_inference_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):
                run_native_bounded(["echo","fake"],cwd=root,env={},
                                   log_path=root/"log",deadline_seconds=99999)
if __name__=="__main__":
    unittest.main()
