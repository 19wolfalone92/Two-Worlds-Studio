"""Unit tests for recliner private-kernel forensic redaction."""
import unittest
from tools.studio_recliner_private_forensic import parse_state, safe_error_line, summarize_logs


class ReclinerForensicTests(unittest.TestCase):
    def test_stage_and_exception_extraction(self):
        result = summarize_logs(
            "WAVE_STAGE_BEGIN_MODEL_LOAD\n"
            "WAVE_STAGE_END_MODEL_LOAD\n"
            "Traceback (most recent call last):\n"
            '  File "/kaggle/working/kernel_main.py", line 18, in run\n'
            "RuntimeError: CUDA out of memory. Tried to allocate 512 MiB\n"
        )
        self.assertEqual(result["last_stage"], "END_MODEL_LOAD")
        self.assertIn("RuntimeError", result["exception_classes"])
        self.assertTrue(result["error_markers"]["cuda_oom"])
        self.assertEqual(result["traceback_functions"], ["run"])
        self.assertTrue(any("CUDA out of memory" in line for line in result["safe_error_lines"]))

    def test_private_prompt_and_path_are_not_returned(self):
        self.assertIsNone(safe_error_line(
            "Photorealistic prompt for the man in the recliner: private actor details"
        ))
        line = safe_error_line("RuntimeError: CUDA error in /kaggle/working/private/model.py")
        self.assertIn("[PRIVATE_PATH]", line)
        self.assertNotIn("/kaggle/", line)

    def test_kernel_state_is_parsed_without_false_completion(self):
        self.assertEqual(parse_state('pavelbloody/kernel has status "KernelWorkerStatus.ERROR"'), "ERROR")
        self.assertEqual(parse_state("no status available"), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
