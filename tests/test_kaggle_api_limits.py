"""Offline tests of Kaggle API dynamic limits and one-attempt-only readback."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from studio.kaggle_api_limits import (
    FILE_PATTERN, KaggleServiceError, _classify_cli_error,
    parse_compute_quota, quota_once, output_once,
)


def process(stdout="", stderr="", code=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=code)


class KaggleApiLimitTests(unittest.TestCase):
    def test_weekly_compute_quota_not_confused_with_undisclosed_api_rate(self):
        raw=json.dumps([
            {"type":"gpu","total":"30.00h","used":"8.06h","remaining":"21.94h",
             "username":"never-show-user","token":"never-show-token"},
            {"type":"tpu","total":"20.00h","used":"0.00h","remaining":"20.00h"},
        ])
        report=parse_compute_quota(raw)
        self.assertEqual(report["compute_quota_rows"][0]["resource"],"GPU")
        self.assertEqual(report["compute_quota_rows"][0]["remaining"],"21.94h")
        self.assertEqual(report["compute_quota_rows"][1]["resource"],"TPU")
        self.assertIsNone(report["kaggle_api_requests_per_day"])
        self.assertIsNone(report["kaggle_api_requests_per_minute"])
        self.assertEqual(report["api_limit_status"],"dynamic_undisclosed")
        self.assertNotIn("never-show",json.dumps(report))

    def test_quota_call_exactly_once(self):
        calls=[]
        def runner(cmd,**kw):
            calls.append(cmd)
            return process(json.dumps([{"remaining":"21.94h","total":"30.00h"}]))
        report=quota_once(runner=runner)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0],["kaggle","quota","--format","json"])
        self.assertEqual(report["status"],"quota_reported")

    def test_429_stop_no_burst_and_no_secret_output(self):
        calls=[]
        def runner(cmd,**kw):
            calls.append(cmd)
            return process("", "HTTP Error 429: Too Many Requests token=not-for-logs", 1)
        with self.assertRaises(KaggleServiceError) as exc:
            quota_once(runner=runner)
        self.assertEqual(len(calls),1)
        self.assertIn("RATE_LIMITED",str(exc.exception))
        self.assertNotIn("not-for-logs",str(exc.exception))

    def test_unexpected_payload_and_unsafe_strings_fail(self):
        with self.assertRaises(KaggleServiceError):
            parse_compute_quota("not json")
        with self.assertRaises(KaggleServiceError):
            parse_compute_quota(json.dumps(["invalid"]))
        with self.assertRaises(KaggleServiceError):
            parse_compute_quota("x" * 150000)
        self.assertEqual(_classify_cli_error("HTTP Error 401: Unauthorised")[0],
                         "authentication_problem")

    def test_single_output_call_filters_model_weights_and_no_force(self):
        calls=[]
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td)
            def runner(cmd,**kwargs):
                calls.append(cmd)
                (directory/"TWO_WORLDS_ARTIFACTS.zip").write_bytes(b"not really a zip")
                (directory/"two_worlds_job_receipt.json").write_text("{}")
                return process()
            result=output_once("sampleuser/two-worlds-comfy-sdxl",directory,runner=runner)
            self.assertEqual(len(calls),1)
            self.assertEqual(result["api_cli_attempts"],1)
            self.assertIn("--file-pattern",calls[0])
            self.assertIn(FILE_PATTERN,calls[0])
            self.assertIn("--page-size",calls[0])
            self.assertNotIn("--force",calls[0])
            self.assertFalse(result["image_verified"])
            self.assertFalse(result["gpu_spent"])

    def test_rejects_download_without_two_proof_files(self):
        with tempfile.TemporaryDirectory() as td:
            def runner(cmd,**kwargs):
                return process()
            with self.assertRaisesRegex(KaggleServiceError,"INCOMPLETE"):
                output_once("example/kernel",Path(td),runner=runner)

    def test_rejects_unsafe_kernel_without_invoking_cli(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                output_once("../../etc/passwd",Path(td),
                            runner=lambda *a,**k: self.fail("CLI should not be invoked"))

    def test_429_output_never_retries(self):
        calls=[]
        with tempfile.TemporaryDirectory() as td:
            def runner(cmd,**kwargs):
                calls.append(cmd)
                return process("", "429 rate limited",code=1)
            with self.assertRaisesRegex(KaggleServiceError,"RATE_LIMITED"):
                output_once("example/kernel",Path(td),runner=runner)
        self.assertEqual(len(calls),1)


if __name__=="__main__":
    unittest.main()
