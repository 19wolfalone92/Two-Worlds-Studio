"""No-network verified readback and cross-account ACL negative-control tests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import eb01_private_two_account_mirror_readback as mod


class ReadbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.accounts = [
            ("AUTHENTICATED_READ_ONLY", "first-person"),
            ("AUTHENTICATED_READ_ONLY", "second-person"),
        ]

    def test_nonsecret_unique_runid_scopes_destination(self):
        ref = mod.mirror_id("second-person", "43898617238")
        self.assertTrue(ref.startswith("second-person/eb01-private-studio-inputs-v1-"))
        self.assertEqual(ref, mod.mirror_id("second-person", "43898617238"))
        self.assertNotEqual(ref, mod.mirror_id("second-person", "43898617239"))
        with self.assertRaises(mod.SourceSyncBlocked):
            mod.mirror_id("second-person", "unknown")

    def test_accessible_for_second_denied_for_first_and_eight_sha_readbacks(self):
        def fake_api(token, args, timeout=0):
            if args[2] == "files" and token == "first-secret":
                raise mod.SourceSyncBlocked("ACCESS_DENIED")
            if args[2] == "download":
                folder = Path(args[args.index("-p") + 1])
                (folder / "media.zip").write_bytes(b"fake-encrypted-for-unit-test")
            return "OK"

        with patch.object(mod, "query", side_effect=self.accounts), \
             patch.object(mod, "kaggle", side_effect=fake_api) as api, \
             patch.object(mod, "validated_extract",
                          return_value={"sources_sha_verified": 8}):
            result = mod.verify_readback(
                "first-secret", "second-secret", "43898617238", self.root)
        self.assertEqual(result["status"],
                         "SECOND_PRIVATE_ACL_AND_ALL_EIGHT_SOURCE_SHAS_VERIFIED")
        self.assertEqual(result["second_source_files_sha_verified"], 8)
        self.assertTrue(result["private_acl_negative_control_pass"])
        self.assertFalse(result["source_files_exposed_to_public_github"])
        self.assertFalse(result["gpu_submitted"])
        self.assertEqual(api.call_count, 3)
        self.assertNotIn("second-person", str(result))

    def test_wrong_public_visibility_blocks_before_download(self):
        with patch.object(mod, "query", side_effect=self.accounts), \
             patch.object(mod, "kaggle", return_value="OK") as api:
            result = mod.verify_readback(
                "first-secret", "second-secret", "43898617238", self.root)
        self.assertEqual(result["status"],
                         "BLOCKED_MIRROR_READABLE_WITH_UNINVITED_FIRST_ACCOUNT")
        self.assertEqual(api.call_count, 2)
        self.assertFalse(result["private_acl_negative_control_pass"])

    def test_403_for_second_blocks_without_transmitting_source(self):
        with patch.object(mod, "query", side_effect=self.accounts), \
             patch.object(mod, "kaggle",
                          side_effect=mod.SourceSyncBlocked("ACCESS_DENIED")) as api:
            result = mod.verify_readback(
                "first-secret", "second-secret", "43898617238", self.root)
        self.assertEqual(result["status"], "BLOCKED_ACCESS_DENIED")
        self.assertEqual(api.call_count, 1)
        self.assertEqual(result["second_source_files_sha_verified"], 0)

    def test_unknown_first_account_failure_never_equals_private_proof(self):
        calls = []
        def fake_api(token, args, timeout=0):
            calls.append(token)
            if token == "first-secret":
                raise mod.SourceSyncBlocked("RATE_LIMITED")
            return "OK"
        with patch.object(mod, "query", side_effect=self.accounts), \
             patch.object(mod, "kaggle", side_effect=fake_api):
            result = mod.verify_readback(
                "first-secret", "second-secret", "43898617238", self.root)
        self.assertEqual(result["status"], "BLOCKED_NEGATIVE_ACL_RATE_LIMITED")
        self.assertEqual(result["second_source_files_sha_verified"], 0)


if __name__ == "__main__":
    unittest.main()
