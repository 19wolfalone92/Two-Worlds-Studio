"""Private original Kaggle dataset archival security tests; ZERO API calls."""
from __future__ import annotations
import os
import json
import hashlib
import zipfile
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
from tools.eb01_archive_verified_private_pilot import (
    private_dataset_meta, archive, archive_existing_output, safe_cmd,
    safe_cmd_result, verify_remote_dataset, PrivateArchiveBlocked
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

    def test_independent_kernel_output_fetch_is_disabled_even_with_owner_token(self):
        with patch.dict(os.environ,{"KAGGLE_API_TOKEN":"owner-test-token"}):
            with patch("tools.eb01_archive_verified_private_pilot.safe_cmd",
                       side_effect=AssertionError("No second Kaggle output request allowed")):
                with tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaisesRegex(PrivateArchiveBlocked,
                            "Separate Kaggle kernel-output downloads are disabled"):
                        archive(KERNEL,SHA,Path(tmp))

    def test_metadata_403_is_not_misclassified_as_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("tools.eb01_archive_verified_private_pilot.safe_cmd_result",
                       return_value=(False,"ACCESS_DENIED")):
                result=verify_remote_dataset(private_dataset_meta(KERNEL,SHA),SHA,Path(tmp))
        self.assertIsNone(result["exists"])
        self.assertEqual(result["status"],"REMOTE_DATASET_LOOKUP_ACCESS_DENIED")

    def test_metadata_lookup_failure_never_starts_dataset_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner=Path(tmp)
            output=runner/"eb01-real-kaggle-supervisor"/"private_download"
            output.mkdir(parents=True)
            movie_bytes=b"owner-private-original-pilot"
            expected=hashlib.sha256(movie_bytes).hexdigest()
            (output/"S02_PILOT33.mp4").write_bytes(movie_bytes)
            (output/"EB01_S02_LTX_PILOT33_JOB.json").write_text(json.dumps({
                "success":True,"clip_sha256":expected,"whole_movie_ready":False}))
            source={"original_pilot_sha256":expected,"decoded_true_neural_frames":33,
                    "artifact_scan_status":"NO_NEON_ARTIFACT_FOUND",
                    "narrow_neon_suspected_frames":[],
                    "native_luma_motion_proxy":{"mean_frame_luma_delta":0.66},
                    "full_human_artistic_approval":False}
            with patch.dict(os.environ,{"KAGGLE_API_TOKEN":"owner-test-token"}):
                with patch("tools.eb01_archive_verified_private_pilot.validate_actual_pilot",
                           return_value=source):
                    with patch("tools.eb01_archive_verified_private_pilot.safe_cmd_result",
                               return_value=(False,"ACCESS_DENIED")):
                        with patch("tools.eb01_archive_verified_private_pilot.safe_cmd",
                                   side_effect=AssertionError("Create/download must not run")):
                            report=archive_existing_output(KERNEL,expected,output,runner)
            self.assertEqual(report["status"],"REMOTE_DATASET_LOOKUP_ACCESS_DENIED")
            self.assertTrue(report["archive_create_blocked"])
            self.assertFalse(report["independent_remote_acl"])
            self.assertFalse(report["remote_original_sha_verified"])

    def test_explicit_http_404_is_the_only_metadata_failure_that_allows_create(self):
        from types import SimpleNamespace
        with patch("tools.eb01_archive_verified_private_pilot.subprocess.run",
                   return_value=SimpleNamespace(returncode=1,stdout="",
                                                 stderr="404 Client Error: Not Found")):
            self.assertEqual(safe_cmd_result(["kaggle","datasets","metadata","x"]),
                             (False,"NOT_FOUND"))
        with patch("tools.eb01_archive_verified_private_pilot.subprocess.run",
                   return_value=SimpleNamespace(returncode=1,stdout="",
                                                 stderr="403 Forbidden")):
            self.assertEqual(safe_cmd_result(["kaggle","datasets","metadata","x"]),
                             (False,"ACCESS_DENIED"))

    def test_only_minimal_receipt_and_mp4_staged_in_private_dataset(self):
        source=Path(__file__).resolve().parents[1].joinpath(
            "tools","eb01_archive_verified_private_pilot.py").read_text()
        self.assertIn('"S02_ORIGINAL_33F_OWNER_REVIEW.mp4"',source)
        self.assertIn('"ORIGINAL_RECEIPT_SAFE.json"',source)
        self.assertNotIn("actions/upload-artifact",source)
        self.assertIn('"isPrivate":True',source)
        self.assertIn('report["independent_remote_acl"]',source)
        self.assertIn('safe["owner_original_mp4_sha256"]=expected',source)
        self.assertIn('validate_actual_pilot(output)',source)


    def test_archive_reuses_same_watchdog_download_and_verifies_private_remote_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner=Path(tmp)
            output=runner/"eb01-real-kaggle-supervisor"/"private_download"
            output.mkdir(parents=True)
            movie_bytes=b"private-original-pilot-test-bytes"
            expected=hashlib.sha256(movie_bytes).hexdigest()
            (output/"S02_PILOT33.mp4").write_bytes(movie_bytes)
            (output/"EB01_S02_LTX_PILOT33_JOB.json").write_text(json.dumps({
                "success":True,"clip_sha256":expected,"whole_movie_ready":False,
                "frame_count":33,"fps":24,"width":320,"height":576}))
            calls=[]
            metadata_calls={"count":0}
            def fake_metadata_result(argv,timeout=180):
                calls.append(list(argv))
                if argv[1:3]!=["datasets","metadata"]:
                    raise AssertionError("Unexpected metadata command")
                metadata_calls["count"]+=1
                if metadata_calls["count"]==1:
                    return False,"NOT_FOUND"
                dest=Path(argv[argv.index("-p")+1])
                dest.mkdir(parents=True,exist_ok=True)
                (dest/"dataset-metadata.json").write_text(
                    json.dumps({"isPrivate":True}),encoding="utf-8")
                return True,"OK"
            def fake_cmd(argv,timeout=180):
                calls.append(list(argv))
                if argv[1:3]==["datasets","create"]:
                    return True
                if argv[1:3]==["datasets","download"]:
                    dest=Path(argv[argv.index("-p")+1])
                    dest.mkdir(parents=True,exist_ok=True)
                    with zipfile.ZipFile(dest/"owner-private.zip","w") as z:
                        z.writestr("S02_ORIGINAL_33F_OWNER_REVIEW.mp4",movie_bytes)
                        z.writestr("ORIGINAL_RECEIPT_SAFE.json",
                                   json.dumps({"owner_original_mp4_sha256":expected}))
                    return True
                raise AssertionError("Unexpected Kaggle command in same-run archive")
            source={"original_pilot_sha256":expected,"decoded_true_neural_frames":33,
                    "artifact_scan_status":"NO_NEON_ARTIFACT_FOUND",
                    "narrow_neon_suspected_frames":[],
                    "native_luma_motion_proxy":{"mean_frame_luma_delta":0.66},
                    "full_human_artistic_approval":False}
            with patch.dict(os.environ,{"KAGGLE_API_TOKEN":"owner-test-token"}):
                with patch("tools.eb01_archive_verified_private_pilot.safe_cmd",side_effect=fake_cmd):
                    with patch("tools.eb01_archive_verified_private_pilot.safe_cmd_result",side_effect=fake_metadata_result):
                        with patch("tools.eb01_archive_verified_private_pilot.validate_actual_pilot",
                                   return_value=source):
                            report=archive_existing_output(KERNEL,expected,output,runner)
            self.assertEqual(report["status"],
                "OWNER_PRIVATE_DATASET_CREATED_ACL_AND_SHA_VERIFIED")
            self.assertTrue(report["independent_remote_acl"])
            self.assertTrue(report["remote_original_sha_verified"])
            self.assertFalse(report["published"])
            self.assertFalse(report["full_movie_ready"])
            self.assertFalse(any(x[1:3]==["kernels","output"] for x in calls))
            self.assertTrue(any(x[1:3]==["datasets","create"] for x in calls))
            # A later hourly run sees the same private dataset, verifies the
            # exact remote MP4 SHA, and does not attempt a duplicate create.
            calls.clear()
            metadata_calls["count"]=1
            with patch.dict(os.environ,{"KAGGLE_API_TOKEN":"owner-test-token"}):
                with patch("tools.eb01_archive_verified_private_pilot.safe_cmd",side_effect=fake_cmd):
                    with patch("tools.eb01_archive_verified_private_pilot.safe_cmd_result",side_effect=fake_metadata_result):
                        with patch("tools.eb01_archive_verified_private_pilot.validate_actual_pilot",
                                   return_value=source):
                            again=archive_existing_output(KERNEL,expected,output,runner)
            self.assertEqual(again["status"],"OWNER_PRIVATE_DATASET_ACL_AND_SHA_VERIFIED")
            self.assertTrue(again["existing_dataset_reused"])
            self.assertFalse(any(x[1:3]==["datasets","create"] for x in calls))

    def test_existing_watchdog_output_must_be_inside_ephemeral_runner_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            runner=base/"runner"
            runner.mkdir()
            outside=base/"outside"
            outside.mkdir()
            with patch.dict(os.environ,{"KAGGLE_API_TOKEN":"owner-test-token"}):
                with self.assertRaisesRegex(PrivateArchiveBlocked,"under ephemeral RUNNER_TEMP"):
                    archive_existing_output(KERNEL,"a"*64,outside,runner)

    def test_existing_watchdog_output_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            runner=base/"runner"
            runner.mkdir()
            real=runner/"real"
            real.mkdir()
            alias=runner/"alias"
            try:
                alias.symlink_to(real,target_is_directory=True)
            except (OSError,NotImplementedError):
                self.skipTest("Directory symlinks unavailable")
            with patch.dict(os.environ,{"KAGGLE_API_TOKEN":"owner-test-token"}):
                with self.assertRaisesRegex(PrivateArchiveBlocked,"Symlinked"):
                    archive_existing_output(KERNEL,"a"*64,alias,runner)

if __name__=="__main__":
    unittest.main()
