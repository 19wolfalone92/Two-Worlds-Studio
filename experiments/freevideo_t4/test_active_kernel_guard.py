"""Regression tests for the durable private-kernel submission lock."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_private_job as job


class ActiveKernelGuardTests(unittest.TestCase):
    def write_lock(self, root: Path, payload: dict) -> Path:
        path = root / "active_kernels.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_missing_lock_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(job.KaggleBlocked, "ACTIVE_KERNEL_LOCK_MISSING_OR_UNSAFE"):
                job.guard_active_kernels(Path(tmp) / "missing.json")

    def test_active_or_unknown_kernel_blocks_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_lock(Path(tmp), {
                "schema": 1, "status": "ACTIVE_OR_STATUS_UNKNOWN",
                "do_not_submit_new_gpu_job": True,
                "submitted_jobs": [{"last_confirmed_status": "QUEUED", "result_recovered": False}],
            })
            with self.assertRaisesRegex(job.KaggleBlocked, "ACTIVE_PRIVATE_KERNELS_UNRESOLVED"):
                job.guard_active_kernels(path)

    def test_terminal_kernel_without_recovered_output_still_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_lock(Path(tmp), {
                "schema": 1, "status": "ALL_TERMINAL",
                "do_not_submit_new_gpu_job": False,
                "submitted_jobs": [{"last_confirmed_status": "COMPLETE", "result_recovered": False}],
            })
            with self.assertRaisesRegex(job.KaggleBlocked, "ACTIVE_PRIVATE_KERNELS_UNRESOLVED"):
                job.guard_active_kernels(path)

    def test_empty_resolved_lock_allows_new_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_lock(Path(tmp), {
                "schema": 1, "status": "NO_ACTIVE_KERNELS",
                "do_not_submit_new_gpu_job": False, "submitted_jobs": [],
            })
            self.assertEqual(job.guard_active_kernels(path),
                             {"status": "NO_ACTIVE_KERNELS_CONFIRMED", "jobs_checked": 0})

    def test_recovered_terminal_kernels_allow_new_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_lock(Path(tmp), {
                "schema": 1, "status": "RESOLVED",
                "do_not_submit_new_gpu_job": False,
                "submitted_jobs": [
                    {"last_confirmed_status": "COMPLETE", "result_recovered": True},
                    {"last_confirmed_status": "ERROR", "result_abandoned": True},
                    {"last_confirmed_status": "CANCEL_ACKNOWLEDGED", "result_recovered": True},
                ],
            })
            self.assertEqual(job.guard_active_kernels(path),
                             {"status": "NO_ACTIVE_KERNELS_CONFIRMED", "jobs_checked": 3})

    def test_invalid_lock_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "active_kernels.json"
            path.write_text("{bad", encoding="utf-8")
            with self.assertRaisesRegex(job.KaggleBlocked, "ACTIVE_KERNEL_LOCK_INVALID"):
                job.guard_active_kernels(path)


if __name__ == "__main__":
    unittest.main()
