"""Offline credential safety tests; never exercise real Kaggle API."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_auth_adapter as adapter


class KaggleAuthAdapterTests(unittest.TestCase):
    def test_modern_access_token_passes_through_unchanged(self):
        format_id, env = adapter.token_environment({
            "KAGGLE_API_TOKEN": "kgt_123456",
            "UNRELATED": "private"})
        self.assertEqual(format_id, "modern_access_token")
        self.assertEqual(env["KAGGLE_API_TOKEN"], "kgt_123456")
        self.assertEqual(env["UNRELATED"], "private")

    def test_legacy_json_is_converted_only_in_memory(self):
        secret = json.dumps({"username": "safe_user", "key": "12345678secret"})
        original = {"KAGGLE_API_TOKEN": secret}
        typ, env = adapter.token_environment(original)
        self.assertEqual(typ, "legacy_json")
        self.assertEqual(original["KAGGLE_API_TOKEN"], secret)
        self.assertEqual(env["KAGGLE_USERNAME"], "safe_user")
        self.assertEqual(env["KAGGLE_KEY"], "12345678secret")
        self.assertNotIn("KAGGLE_API_TOKEN", env)

    def test_malformed_legacy_secret_refused(self):
        for raw in ("{bad", json.dumps({"username": "test", "key": "secret", "extra": "x"}),
                    json.dumps({"username": "../../evil", "key": "secretAAAA"})):
            with self.subTest(raw=raw[:6]), self.assertRaises(adapter.KaggleAuthBlocked):
                adapter.token_environment({"KAGGLE_API_TOKEN": raw})

    def test_redaction_of_unknown_api_output(self):
        secret = "DO_NOT_PRINT_THIS_PRIVATE_CREDENTIAL"
        def fake_runner(cmd, *, env, text, capture_output, timeout, check):
            self.assertEqual(cmd, ["kaggle", "quota", "--format", "json"])
            return SimpleNamespace(returncode=1, stdout="", stderr=secret + " code 401")
        with self.assertRaisesRegex(adapter.KaggleAuthBlocked, "^KAGGLE_unauthorized$") as err:
            adapter.check_quota(env={"KAGGLE_API_TOKEN": "dummy"}, runner=fake_runner)
        self.assertNotIn(secret, str(err.exception))

    def test_successful_mocked_quota_is_read_only(self):
        def fake_runner(cmd, *, env, text, capture_output, timeout, check):
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps([
                {"name":"GPU","remaining":"13h"}, {"name":"TPU","remaining":"0h"}]))
        result = adapter.check_quota(env={"KAGGLE_API_TOKEN":"dummy"},
                                     runner=fake_runner)
        self.assertEqual(result["status"], "API_AUTHENTICATED_QUOTA_READ_ONLY")
        self.assertEqual(result["gpu_quota_rows"], 1)
        self.assertFalse(result["gpu_launched"])

    def test_cli_error_classification_is_redacted(self):
        cases = [
            ("usage: kaggle k list: error: unrecognized arguments: --format json",
             "cli_incompatible"),
            ("Traceback (most recent call last):\\nRuntimeError: secret_message",
             "python_exception_redacted"),
            ("SSLERROR certificate verify failed", "ssl_failure"),
        ]
        for raw, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(adapter.diagnose_error(raw), expected)

    def test_kernel_list_auth_error_does_not_leak_cli_output(self):
        secret = "PRIVATE_KEY_DO_NOT_LEAK"
        def fake_runner(cmd, *, env, text, capture_output, timeout, check):
            self.assertEqual(cmd, ["kaggle", "kernels", "list", "--mine", "--page-size", "1"])
            return SimpleNamespace(returncode=1, stderr=secret + "\\nUnauthorized (401)", stdout="")
        with self.assertRaisesRegex(adapter.KaggleAuthBlocked, "^KAGGLE_unauthorized$") as ctx:
            adapter.check_kernels_list(env={"KAGGLE_API_TOKEN":"placeholder"}, runner=fake_runner)
        self.assertNotIn(secret, str(ctx.exception))

    def test_sanitized_private_kernel_list_success(self):
        """Private notebook names never enter return values or logs."""
        secret = "PRIVATE_SHOULD_NOT_LEAK"
        def fake_runner(cmd, *, env, text, capture_output, timeout, check):
            self.assertEqual(cmd, ["kaggle", "kernels", "list", "--mine",
                                   "--page-size", "1"])
            return SimpleNamespace(returncode=0, stderr="",
                                   stdout="ref | title\\nPRIVATE_SHOULD_NOT_LEAK | " + secret)
        result = adapter.check_kernels_list(
            env={"KAGGLE_API_TOKEN": "dummy"}, runner=fake_runner)
        self.assertEqual(result["status"], "API_AUTHENTICATED_PRIVATE_KERNELS_READ_ONLY")
        self.assertNotIn(secret, json.dumps(result))
        self.assertFalse(result["gpu_launched"])



if __name__ == "__main__":
    unittest.main()
