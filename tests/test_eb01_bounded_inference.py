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
    def test_invalid_unbounded_inference_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):
                run_native_bounded(["echo","fake"],cwd=root,env={},
                                   log_path=root/"log",deadline_seconds=99999)
if __name__=="__main__":
    unittest.main()
