"""CPU release tests are synthetic, never connect Kaggle or publish."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from tools.batch03_cpu_release import (
    SOURCE_SHA, ReleaseBlocked, index_private_outputs, safe_package, source_location, release
)


class CpuReviewRelayTests(unittest.TestCase):
    def test_selects_only_approved_latest_kernel_for_each_kind(self):
        handles=["wolf19loner92/tw-batch03-new-20261104-01",
                 "wolf19loner92/tw-batch03-old-20261104-01"]
        def fake(*cmd, **kwargs):
            handle=cmd[2]
            return ("fileName\nTwoWorlds_PRIVATE_13_image.zip\n"
                    if "new" in handle else
                    "fileName\nTwoWorlds_PRIVATE_13_image.zip\n"
                    "TwoWorlds_PRIVATE_13_voice.zip\n")
        with patch("tools.batch03_cpu_release.list_owned_kernels", return_value=(handles,True)):
            with patch("tools.batch03_cpu_release.command", side_effect=fake):
                mapping=index_private_outputs()
        self.assertEqual(mapping["13","image"],handles[0])
        self.assertEqual(mapping["13","voice"],handles[1])

    def test_refuses_modified_original_source(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/"batch03_source.zip").write_bytes(b"wrong source")
            with self.assertRaises(ReleaseBlocked):
                source_location(folder)

    def test_private_package_only_contains_finished_review_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/"work_13";p.mkdir()
            for name in ("final.mp4","cover.jpg","subtitles.srt",
                         "qa_report.json","receipt.json"):
                (p/name).write_bytes(b"test-only")
            (p/"story.txt").write_text("PRIVATE SOURCE MUST NOT BE INCLUDED")
            result=root/"encrypted_input.zip"
            safe_package(root,result)
            with zipfile.ZipFile(result) as archive:
                files=archive.namelist()
            self.assertEqual(len(files),5)
            self.assertTrue(all("work_13" in f for f in files))
            self.assertTrue(all("story.txt" not in f for f in files))

    def test_wrong_mode_never_calls_kaggle(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);req=folder/"request.json"
            req.write_text(json.dumps({"schema":1,"mode":"public_post","free_resources_only":True,
                                        "source_sha256":SOURCE_SHA,
                                        "dataset_slug":"two-worlds-private-batch03-20261104-repair-v6",
                                        "nonce":"safe-check-20261104"}))
            with patch("tools.batch03_cpu_release.command") as runner:
                with self.assertRaises(ReleaseBlocked):
                    release(req, folder/"private_out")
                runner.assert_not_called()
