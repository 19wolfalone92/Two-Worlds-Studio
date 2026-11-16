"""Read-only fake-CLI test harness: no Kaggle traffic, no account leakage."""
import subprocess
import unittest

from tools.eb01_dual_kaggle_readiness import evaluate


def fake_cli(args, **kwargs):
    token = kwargs.get("env", {}).get("KAGGLE_API_TOKEN")
    if not token:
        raise AssertionError("Runner must scope exactly one account token")
    handle = "account_a" if token == "test-first" else "account_b"
    if args[:3] == ["kaggle", "config", "view"]:
        return subprocess.CompletedProcess(args, 0,
                                           stdout="username: " + handle + "\n",
                                           stderr="")
    if args[:3] == ["kaggle", "kernels", "list"]:
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")
    if args[:3] == ["kaggle", "datasets", "files"]:
        allowed = token == "test-first"
        return subprocess.CompletedProcess(
            args, 0 if allowed else 1,
            stdout="originals available" if allowed else "",
            stderr="" if allowed else "403 forbidden")
    raise AssertionError("A non-read-only Kaggle command was requested")


def both_access_cli(args, **kwargs):
    result = fake_cli(args, **kwargs)
    if args[:3] == ["kaggle", "datasets", "files"]:
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")
    return result


class KaggleDualAccessAuditTests(unittest.TestCase):
    def test_actual_second_account_acl_denied_disables_gpu_and_no_source_copy(self):
        result = evaluate("test-first", "test-second", runner=fake_cli)
        self.assertTrue(result["two_distinct_authenticated_accounts"])
        self.assertFalse(result["private_dataset_shared_with_second_account"])
        self.assertEqual(result["hypothetical_balanced_scene_split_if_BOTH_are_ready"],
                         {"owner": 15, "collaborator": 15})
        self.assertIn("PRIVATE_CANONICAL_SOURCE_ACCESS_NOT_VERIFIED",
                      result["account_blocking_gates"]["collaborator"])
        self.assertEqual(result["next_gpu_take_proposals"], [])
        self.assertEqual(result["gpu_jobs_submitted"], 0)
        self.assertFalse(result["movie_ready"])
        self.assertFalse(result["private_original_media_transferred"])
        serialized = str(result)
        self.assertNotIn("account_a", serialized)
        self.assertNotIn("account_b", serialized)
        self.assertNotIn("test-first", serialized)

    def test_both_accounts_with_source_metadata_still_require_native_pilot(self):
        result = evaluate("test-first", "test-second", runner=both_access_cli)
        self.assertTrue(result["private_dataset_shared_with_second_account"])
        self.assertIn("NATIVE_576X1024_HUMAN_REVIEWED_PILOT_MISSING",
                      result["account_blocking_gates"]["owner"])
        self.assertEqual(result["next_gpu_take_proposals"], [])
        self.assertFalse(result["native_quality_pilot_approved"])

    def test_same_token_does_not_fake_two_distinct_accounts(self):
        result = evaluate("test-first", "test-first", runner=both_access_cli)
        self.assertFalse(result["two_distinct_authenticated_accounts"])
        self.assertEqual(result["second_source_metadata"],
                         "DISTINCT_AUTH_NOT_VERIFIED")


if __name__ == "__main__":
    unittest.main()
