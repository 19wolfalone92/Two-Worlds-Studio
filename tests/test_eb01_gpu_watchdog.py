"""GPU observer tests: queued resource is never a failed video model.

All tests offline. No network, no actual GPU, no credit consumption.
"""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from tools.eb01_gpu_watchdog import (
    KERNEL, parse_state, classify_failure, sanitize, inspect
)


def fake_runner(state, output_code=0):
    def response(args, timeout=90):
        if args[2] == "status":
            return (0, f"{KERNEL} has status \"KernelWorkerStatus.{state}\"")
        if args[2] == "output":
            return (output_code, "Kaggle output downloaded" if not output_code else "404")
        if args[2] == "logs":
            return (0, "RuntimeError: synthetic CUDA out of memory")
        raise AssertionError("No GPU push, new job or service permission")
    return response


class KaggleGPUWatchdogTests(unittest.TestCase):
    def test_correctly_parse_queued_running_error_complete(self):
        for state in ("QUEUED","RUNNING","ERROR","CANCEL","COMPLETE"):
            self.assertEqual(parse_state(f'status "KernelWorkerStatus.{state}"'),state)
        self.assertEqual(parse_state("job has pending status"),"UNKNOWN")

    def test_queued_not_called_failed_model_or_new_gpu_job(self):
        with tempfile.TemporaryDirectory() as temp:
            r=inspect(KERNEL,Path(temp),runner=fake_runner("QUEUED"))
        self.assertEqual(r["status"],"WAITING_FOR_FREE_KAGGLE_GPU")
        self.assertEqual(r["observations"],["QUEUED"])
        self.assertFalse(r["gpu_job_submitted"])
        self.assertFalse(r["full_movie_ready"])

    def test_running_is_bounded_without_gpu_resubmit(self):
        with tempfile.TemporaryDirectory() as temp:
            r=inspect(KERNEL,Path(temp),max_polls=3,poll_seconds=0,
                      runner=fake_runner("RUNNING"),sleeper=lambda x:None)
        self.assertEqual(r["observations"],["RUNNING"]*3)
        self.assertEqual(r["status"],"INFERENCE_IN_PROGRESS_NO_RESULT_YET")
        self.assertFalse(r["gpu_job_submitted"])

    def test_error_is_diagnosed_then_stopped_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            r=inspect(KERNEL,Path(temp),runner=fake_runner("ERROR"))
        self.assertEqual(r["status"],"REAL_GPU_INFERENCE_ERROR_DIAGNOSED")
        self.assertEqual(r["failure_category"],"CUDA_VRAM_EXHAUSTION")
        self.assertEqual(r["next_action"],"ROOT_CAUSE_PATCH_AND_TEST_FIRST_NO_AUTORETRY")

    def test_complete_without_actual_mp4_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            r=inspect(KERNEL,Path(temp),runner=fake_runner("COMPLETE"))
        self.assertEqual(r["status"],"KERNEL_COMPLETE_BUT_NO_VALID_TRUE_VIDEO")
        self.assertFalse(r["full_movie_ready"])

    def test_stale_credentials_are_not_interpreted_as_running(self):
        def bad(args,timeout=90):
            return 1,"401 unauthorized"
        with tempfile.TemporaryDirectory() as temp:
            r=inspect(KERNEL,Path(temp),runner=bad)
        self.assertEqual(r["status"],"KAGGLE_STATUS_UNREADABLE")

    def test_refuse_arbitrary_kernel_and_unbounded_watches(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                inspect("other-user/random",Path(temp))
            with self.assertRaises(ValueError):
                inspect(KERNEL,Path(temp),max_polls=100)

    def test_real_ltx_33frame_timeout_is_classified(self):
        self.assertEqual(classify_failure("RuntimeError: PILOT inference timed out after 1400 seconds",None),
                         "INFERENCE_TIMEOUT_NEEDS_STAGE_TIMING")
        self.assertEqual(classify_failure(
            "float16 operations on cpu are unsupported, subprocess.TimeoutExpired after 1400s",None),
            "INFERENCE_TIMEOUT_WITH_UNSUPPORTED_FP16_CPU_PATH")

    def test_private_secrets_and_source_paths_redacted(self):
        logs=("Traceback private /kaggle/input/wolf/ELIZABETH.png\n"
              "ERROR Authorization:BearerABC hf_12345678901234567890")
        red=sanitize(logs)
        self.assertNotIn("/kaggle/input/wolf",str(red))
        self.assertNotIn("hf_123",str(red))

if __name__=="__main__":
    unittest.main()
