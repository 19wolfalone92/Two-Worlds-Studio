"""Synthetic no-network tests: privacy constraints, hashes, and disjoint tokens."""
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools import eb01_private_two_account_dataset_sync as mod

SAMPLES = {
    "KOSTYA.png": b"synthetic-skeleton-image-1234",
    "ELIZABETH.png": b"synthetic-interviewer-image-1234",
    "S01.wav": b"synthetic-voice-one-1234",
}


def sample_archive(path, samples=SAMPLES):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for k, v in samples.items():
            z.writestr("private/source/" + k, v)


class ArchiveSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.zip = self.root / "source.zip"
        sample_archive(self.zip)
        self.expected = {k: hashlib.sha256(v).hexdigest()
                         for k, v in SAMPLES.items()}

    def test_extract_only_expected_bytes_and_verify_sha(self):
        result = mod.validated_extract(self.zip, self.root / "out", self.expected)
        self.assertEqual(result["sources_sha_verified"], 3)
        self.assertEqual({p.name for p in (self.root / "out").iterdir()},
                         set(self.expected))
        for filename in self.expected:
            self.assertEqual(mod.sha256(self.root / "out" / filename),
                             self.expected[filename])

    def test_reject_missing_source_and_wrong_hash(self):
        with self.assertRaises(mod.SourceSyncBlocked):
            mod.validated_extract(self.zip, self.root / "missing",
                                  {**self.expected, "S02.wav": "0"*64})
        with self.assertRaises(mod.SourceSyncBlocked):
            mod.validated_extract(self.zip, self.root / "wrong",
                                  {**self.expected, "KOSTYA.png": "0"*64})

    def test_reject_duplicate_canonical_filename(self):
        with zipfile.ZipFile(self.zip, "a") as z:
            z.writestr("private/duplicate/KOSTYA.png", SAMPLES["KOSTYA.png"])
        with self.assertRaisesRegex(mod.SourceSyncBlocked, "AMBIGUOUS"):
            mod.validated_extract(self.zip, self.root / "duplicate", self.expected)

    def test_reject_path_escape_even_for_unselected_entries(self):
        with zipfile.ZipFile(self.zip, "a") as z:
            z.writestr("../outside.txt", "unsafe")
        with self.assertRaisesRegex(mod.SourceSyncBlocked, "UNSAFE"):
            mod.validated_extract(self.zip, self.root / "traversal", self.expected)

    def test_private_metadata_must_be_explicit_boolean_true(self):
        meta = self.root / "dataset-metadata.json"
        for value in (False, None, "true", 1):
            meta.write_text(json.dumps({"isPrivate": value}))
            self.assertFalse(mod.privacy_metadata(meta))
        meta.write_text(json.dumps({"isPrivate": True}))
        self.assertTrue(mod.privacy_metadata(meta))

    def test_error_classification_prioritizes_permissions(self):
        self.assertEqual(mod.classify_error("403 forbidden, 404 also mentioned"),
                         "ACCESS_DENIED")
        self.assertEqual(mod.classify_error("429 too many requests"),
                         "RATE_LIMITED")
        self.assertEqual(mod.classify_error("unexpected inaccessible server"),
                         "UNCLASSIFIED_CLI_ERROR")


class FullPrivateSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_distinct_accounts_and_access_already_shared_skips_copy(self):
        with patch.object(mod, "query", side_effect=[
            ("AUTHENTICATED_READ_ONLY", "first-person"),
            ("AUTHENTICATED_READ_ONLY", "second-person")
        ]), patch.object(mod, "kaggle", return_value="OK") as api:
            result = mod.perform_one_authorized_private_sync("tok-owner",
                                                               "tok-second",
                                                               self.root)
            self.assertEqual(result["status"],
                             "ORIGINAL_PRIVATE_DATASET_SHARED_NO_COPY")
            self.assertTrue(result["two_distinct_accounts"])
            self.assertFalse(result["second_private_copy_verified"])
            self.assertEqual(api.call_count, 1)

    def test_identical_tokens_block_without_kaggle_use(self):
        with patch.object(mod, "kaggle") as api:
            result = mod.perform_one_authorized_private_sync("duplicate",
                                                               "duplicate",
                                                               self.root)
            self.assertEqual(result["status"], "BLOCKED_TOKENS_MUST_DIFFER")
            api.assert_not_called()

    def test_denied_source_and_unknown_mirror_failure_does_not_copy(self):
        with patch.object(mod, "query", side_effect=[
            ("AUTHENTICATED_READ_ONLY", "first-person"),
            ("AUTHENTICATED_READ_ONLY", "second-person")
        ]), patch.object(mod, "kaggle",
                         side_effect=mod.SourceSyncBlocked("ACCESS_DENIED")) as api, \
             patch.object(mod, "verify_private_mirror",
                          return_value="MIRROR_METADATA_AUTH_DENIED"):
            result = mod.perform_one_authorized_private_sync("owner",
                                                               "other",
                                                               self.root)
            self.assertIn("BLOCKED_MIRROR_METADATA_AUTH_DENIED", result["status"])
            self.assertFalse(result["second_private_copy_verified"])
            self.assertEqual(api.call_count, 1)

    def test_private_fallback_upload_only_exact_canonical_files(self):
        expected = {k: hashlib.sha256(v).hexdigest() for k, v in SAMPLES.items()}
        actions = []

        def fake_cli(token, args, timeout=120):
            actions.append((token, args))
            if args[2] == "files":
                raise mod.SourceSyncBlocked("ACCESS_DENIED")
            if args[2] == "download":
                target = Path(args[args.index("-p") + 1])
                target.mkdir(parents=True, exist_ok=True)
                sample_archive(target / "source.zip")
            elif args[2] == "create":
                stage = Path(args[args.index("-p") + 1])
                self.assertEqual({x.name for x in stage.iterdir()},
                                 set(expected) | {"dataset-metadata.json"})
                self.assertTrue(mod.privacy_metadata(stage / "dataset-metadata.json"))
                for filename, digest in expected.items():
                    self.assertEqual(mod.sha256(stage / filename), digest)
            self.assertNotIn("--public", args)
            return "OK"

        with patch.object(mod, "FILE_SHA", expected), \
             patch.object(mod, "query", side_effect=[
                 ("AUTHENTICATED_READ_ONLY", "first-person"),
                 ("AUTHENTICATED_READ_ONLY", "second-person")]), \
             patch.object(mod, "kaggle", side_effect=fake_cli), \
             patch.object(mod, "verify_private_mirror",
                          side_effect=["MIRROR_ABSENT",
                                       "MIRROR_PRIVATE_SHA_VERIFIED"]):
            result = mod.perform_one_authorized_private_sync("tok-owner",
                                                               "tok-second",
                                                               self.root)
        self.assertEqual(result["status"], "NEW_PRIVATE_MIRROR_SHA_ACL_VERIFIED")
        self.assertTrue(result["second_private_copy_verified"])
        self.assertEqual([cmd[1][2] for cmd in actions],
                         ["files", "download", "create"])
        self.assertEqual(actions[1][0], "tok-owner")
        self.assertEqual(actions[2][0], "tok-second")
        self.assertNotIn("second-person", str(result))
        self.assertEqual(result["gpu_jobs_submitted"], 0)


if __name__ == "__main__":
    unittest.main()
