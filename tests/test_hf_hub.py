"""Hugging Face integration tests (fully offline; no real token)."""
from __future__ import annotations

import io
import json
import unittest
from urllib import error

from studio.hf_hub import HUB_API, HubPreflightError, check_connection, hub_json, planned_report


class FakeHTTPResponse:
    def __init__(self, value):
        self.body = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, n):
        return self.body[:n]


class HuggingFaceHubTests(unittest.TestCase):
    def test_dry_run_does_not_require_token(self):
        report = planned_report()
        self.assertFalse(report["authentication_tested"])
        self.assertFalse(report["weights_downloaded"])
        self.assertEqual({m["role"] for m in report["models"]}, {"image", "voice"})

    def test_readonly_token_header_and_bounded_request(self):
        received = []

        def fake_open(req, timeout):
            received.append(req)
            self.assertEqual(timeout, 15)
            return FakeHTTPResponse({"name": "sample"})

        result = hub_json("/whoami-v2", "hf_test_value", opener=fake_open)
        self.assertEqual(result["name"], "sample")
        self.assertEqual(received[0].full_url, HUB_API + "/whoami-v2")
        self.assertEqual(received[0].get_method(), "GET")
        self.assertEqual(received[0].get_header("Authorization"), "Bearer hf_test_value")

    def test_missing_token_and_invalid_path(self):
        with self.assertRaises(HubPreflightError):
            hub_json("/whoami-v2", "")
        with self.assertRaises(ValueError):
            hub_json("/../elsewhere", "hf_test")

    def test_connection_response_never_exposes_secret_or_identity(self):
        def query(path, token):
            self.assertEqual(token, "hf_super_secret")
            if path == "/whoami-v2":
                return {"name": "real_account_name", "email": "user@example.org"}
            return {"id": path.removeprefix("/models/"), "sha": "f" * 40,
                    "cardData": {"license": "apache-2.0"}, "gated": False}

        result = check_connection("hf_super_secret", query=query)
        printed = json.dumps(result)
        self.assertEqual(len(result["models"]), 2)
        self.assertNotIn("hf_super_secret", printed)
        self.assertNotIn("real_account_name", printed)
        self.assertFalse(result["inference_performed"])
        self.assertFalse(result["gpu_allocation_verified"])

    def test_unreviewed_license_fails_closed(self):
        def query(path, token):
            if path == "/whoami-v2":
                return {"name": "sample"}
            return {"id": path.removeprefix("/models/"), "cardData": {"license": "unknown"}}
        with self.assertRaisesRegex(HubPreflightError, "license"):
            check_connection("hf_fake", query=query)

    def test_http_error_discards_secret_and_response_body(self):
        def fake_open(req, timeout):
            raise error.HTTPError(req.full_url, 403, "hf_test_is_secret", {}, io.BytesIO(b"secret response"))
        with self.assertRaises(HubPreflightError) as caught:
            hub_json("/whoami-v2", "hf_test_is_secret", opener=fake_open)
        self.assertNotIn("hf_test_is_secret", str(caught.exception))
        self.assertIn("403", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
