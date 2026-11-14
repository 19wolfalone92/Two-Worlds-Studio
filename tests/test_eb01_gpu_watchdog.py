"""GPU observer tests: queued resource is never a failed video model.

All tests offline. No network, no actual GPU, no credit consumption.
"""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from tools.eb01_gpu_watchdog import (
    KERNEL, parse_state, classify_failure, sanitize, inspect, private_numeric_qa_summary
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

    def test_real_terminal_t4_330s_generated_mp4_then_color_qa_failed(self):
        # Original GPU logs showed a misleading FP16 CPU warning followed
        # by terminal frame/color QA failure; NEVER mislabel model FP16 crash.
        logs="Pipelines loaded with dtype=torch.float16 cannot run with cpu device"
        receipt={"success":False,"status":"REAL_ALTERNATIVE_GPU_BLOCKED_NO_FALSE_FILM_CLAIM",
                 "error_summary":"Authentic LTX S02 failed required video frame/color QA"}
        self.assertEqual(classify_failure(logs,receipt),
            "GENUINE_GPU_VIDEO_GENERATED_BUT_VISUAL_COLOR_OR_GEOMETRY_QA_FAILED")
        self.assertEqual(classify_failure(
            logs+"\nRuntimeError: Authentic LTX S02 failed required video frame/color QA",None),
            "GENUINE_GPU_VIDEO_GENERATED_BUT_VISUAL_COLOR_OR_GEOMETRY_QA_FAILED")
        self.assertEqual(classify_failure(logs,None),
                         "UNSUPPORTED_CPU_HALF_PRECISION_VAE_OR_ENCODER")

    def test_real_gpu_private_frame_qa_summary_extracted_without_actor_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            receipt={"schema":1,"shot":"S02",
                "rgb_gate_status":"REJECT_PERSISTENT_NARROW_NEON_STRIPES_REVIEW_REQUIRED",
                "frame_flags":[1,6,19],
                "checks":{"native_frames_33":True,"no_neon":False},
                "private_user_prompt":"DO NOT LEAK THIS",
                "six_original_russian_wav_bytes":"DO NOT LEAK WAV"}
            (root/"EB01_S02_NATIVE_GPU_FRAME_QA_SAFE.json").write_text(json.dumps(receipt))
            result=private_numeric_qa_summary(root)
            self.assertEqual(result["source"],"PRIVATE_WORKER_ORIGINAL_QA_RECEIPT")
            self.assertEqual(result["frame_flags"],[1,6,19])
            self.assertFalse(result["checks"]["no_neon"])
            self.assertNotIn("private_user_prompt",str(result))
            self.assertNotIn("six_original_russian_wav_bytes",str(result))

    def test_missing_and_broken_native_qa_never_count_as_good_video(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            result=private_numeric_qa_summary(root)
            self.assertEqual(result["source"],"PRIVATE_GPU_OUTPUT_HAS_NO_NATIVE_VIDEO_OR_QA_RECEIPT")
            (root/"EB01_S02_NATIVE_GPU_FRAME_QA_SAFE.json").write_text("{ invalid json")
            result=private_numeric_qa_summary(root)
            self.assertEqual(result["source"],"INVALID_GPU_QA_RECEIPT")

    def test_private_secrets_and_source_paths_redacted(self):
        logs=("Traceback private /kaggle/input/wolf/ELIZABETH.png\n"
              "ERROR Authorization:BearerABC hf_12345678901234567890")
        red=sanitize(logs)
        self.assertNotIn("/kaggle/input/wolf",str(red))
        self.assertNotIn("hf_123",str(red))

if __name__=="__main__":
    unittest.main()
