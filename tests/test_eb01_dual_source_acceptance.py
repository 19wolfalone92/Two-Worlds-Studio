import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from tools import eb01_dual_source_acceptance as m


class SharedSourceAcceptance(unittest.TestCase):
    def test_listing_alone_cannot_pass_sha_acceptance(self):
        with tempfile.TemporaryDirectory() as t:
            result = m.verify_account("first", Path(t) / "private", api=lambda *a, **kw: None)
            self.assertTrue(result["listing_verified"])
            self.assertFalse(result["download_verified"])
            self.assertEqual(result["canonical_sha_verified"], 0)
            self.assertFalse((Path(t) / "private").exists())

    def test_independent_download_bytes_and_no_other_token(self):
        expected = {"KOSTYA.png": hashlib.sha256(b"synthetic-original-123").hexdigest()}
        calls = []
        def api(token, args, **kw):
            calls.append((token, args[2]))
            if args[2] == "download":
                with zipfile.ZipFile(Path(args[args.index("-p") + 1]) / "source.zip", "w") as z:
                    z.writestr("KOSTYA.png", b"synthetic-original-123")
        with tempfile.TemporaryDirectory() as t, patch.object(m, "FILE_SHA", expected), \
             patch("tools.eb01_private_two_account_dataset_sync.FILE_SHA", expected):
            report = m.accept("first", "second", Path(t), api=api,
                              auth=lambda token: ("AUTHENTICATED_READ_ONLY", token))
        self.assertTrue(report["all_eight_sources_verified_for_both"])
        self.assertEqual(calls, [("first", "files"), ("first", "download"),
                                 ("second", "files"), ("second", "download")])
        self.assertNotIn("first", json.dumps(report))
        self.assertEqual(report["dataset_mutations"], 0)

    def test_denied_second_download_stays_blocked_and_wipes_private_files(self):
        def api(token, args, **kw):
            if token == "second":
                raise m.SourceSyncBlocked("ACCESS_DENIED")
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            report = m.accept("first", "second", root, api=api,
                              auth=lambda token: ("AUTHENTICATED_READ_ONLY", token))
            self.assertTrue(report["accounts"]["collaborator"]["http_403_seen"])
            self.assertFalse(report["all_eight_sources_verified_for_both"])
            self.assertEqual(list(root.iterdir()), [])

    def test_no_network_for_duplicate_token(self):
        with tempfile.TemporaryDirectory() as t:
            report = m.accept("same", "same", Path(t), auth=lambda t: self.fail())
        self.assertFalse(report["two_distinct_authenticated_accounts"])

    def test_child_environment_has_exactly_one_account_secret(self):
        with patch.dict(os.environ, {"EB01_OWNER_TOKEN": "private1", "EB01_SECOND_TOKEN": "private2",
                                     "KAGGLE_KEY": "legacy", "KAGGLE_USERNAME": "wrong",
                                     "GH_TOKEN": "github", "KAGGLE_CONFIG_DIR": "/old"}):
            env = m.isolated_environment("selected", Path("/new"))
        self.assertEqual(env["KAGGLE_API_TOKEN"], "selected")
        self.assertEqual(env["KAGGLE_CONFIG_DIR"], "/new")
        for key in ("EB01_OWNER_TOKEN", "EB01_SECOND_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME", "GH_TOKEN"):
            self.assertNotIn(key, env)


if __name__ == "__main__":
    unittest.main()
