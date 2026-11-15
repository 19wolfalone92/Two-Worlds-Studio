"""Private original Kaggle dataset archival security tests; ZERO API calls."""
from __future__ import annotations
import os
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
from tools.eb01_archive_verified_private_pilot import (
    private_dataset_meta, archive, safe_cmd, PrivateArchiveBlocked
)

KERNEL="wolf19loner92/eb01-ltx2b-pilot33-a609d40bc3ee"
SHA="a"*64

class OwnerPrivateArchivalSecurity(unittest.TestCase):
    def test_private_metadata_default_not_public(self):
        m=private_dataset_meta(KERNEL,SHA)
        self.assertTrue(m["isPrivate"])
        self.assertEqual(m["licenses"],[{"name":"copyright-authors"}])
        self.assertEqual(m["id"],"wolf19loner92/eb01-owner-private-motion-a609d40bc3ee")
        self.assertGreaterEqual(len(m["subtitle"]),20)

    def test_public_upload_flag_rejected_even_before_subprocess(self):
        with self.assertRaisesRegex(PrivateArchiveBlocked,"Never publish"):
            safe_cmd(["kaggle","datasets","create","--public"])

    def test_unpinned_kernel_and_hash_are_blocked(self):
        with self.assertRaises(PrivateArchiveBlocked):
            private_dataset_meta("other/owner-private",SHA)
        with self.assertRaises(PrivateArchiveBlocked):
            private_dataset_meta(KERNEL,"0"*63)

    def test_no_owner_secret_blocks_before_read_or_write(self):
        with patch.dict(os.environ,{},clear=True):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(PrivateArchiveBlocked,"credential"):
                    archive(KERNEL,SHA,Path(tmp))

    def test_only_minimal_receipt_and_mp4_staged_in_private_dataset(self):
        source=Path(__file__).resolve().parents[1].joinpath(
            "tools","eb01_archive_verified_private_pilot.py").read_text()
        self.assertIn('"S02_ORIGINAL_33F_OWNER_REVIEW.mp4"',source)
        self.assertIn('"ORIGINAL_RECEIPT_SAFE.json"',source)
        self.assertNotIn("actions/upload-artifact",source)
        self.assertIn('"isPrivate":True',source)
        self.assertIn('report["independent_remote_acl"]',source)
        self.assertIn('validate_actual_pilot(output)',source)

if __name__=="__main__":
    unittest.main()
