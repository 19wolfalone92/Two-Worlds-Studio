"""CPU integration: genuine model output persists BEFORE any failed artistic QC.

FAKE FFmpeg test patterns here are only test fixtures. They must never be
called an original model clip or uploaded as a finished film. No GPU/API.
"""
from __future__ import annotations
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from studio.comfy_neural import _probe
from studio.eb01_v7_private_checkpoint import (
    CheckpointRejected, _sha, checkpoint_raw, mark_quarantined,
    selected_contact_sheet
)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "Requires actual FFmpeg decoder, never skip in CI")
class E01OriginalWanCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.output = self.folder/"private-working"
        self.reference = self.folder/"KOSTYA-source-test-only.png"
        Image.effect_noise((320,576),25).convert("RGB").save(self.reference)
        self.raw = self.folder/"raw_neural_test_fixture.mp4"
        self.receipt = self.raw.with_suffix(".receipt.json")
        # 21 frame fake video: 17 content frames + four terminal buffer frames.
        subprocess.run([
            "ffmpeg","-hide_banner","-nostdin","-loglevel","error","-y",
            "-f","lavfi","-i","testsrc2=size=320x576:rate=24",
            "-frames:v","21","-an","-c:v","libx264",
            "-pix_fmt","yuv420p",str(self.raw)
        ],check=True,capture_output=True,timeout=60)
        data={
            "schema":1,"engine":"neural_comfy","model":"wan22-5b",
            "inference_performed":True,
            "clip_sha256":_sha(self.raw),
            "reference_sha256":_sha(self.reference),
            "video":_probe(self.raw),
            "workflow_sha256":"f"*64,
            "published":False,
            "test_only_fake_video":True
        }
        self.receipt.write_text(json.dumps(data),encoding="utf-8")

    def test_original_model_output_is_persisted_without_change(self):
        saved = checkpoint_raw(self.raw,self.receipt,self.reference,
                               self.output,"S02",17)
        original = self.output/"S02.original_wan_raw.mp4"
        proof = self.output/"S02.original_wan_raw.receipt.json"
        self.assertEqual(saved["status"],"GENUINE_NEURAL_SOURCE_PRESERVED_QA_PENDING")
        self.assertEqual(saved["original_mp4_sha256"],_sha(self.raw))
        self.assertEqual(_sha(original),_sha(self.raw))
        self.assertEqual(_sha(proof),_sha(self.receipt))
        self.assertEqual(saved["raw_frames"],21)
        self.assertFalse(saved["can_use_in_finished_episode"])
        self.assertFalse(saved["published"])
        # Risky QC must not delete or overwrite the original.
        report = mark_quarantined(self.output,"S02",
                  "Automated neon heuristic needs human review",
                  inspection={"flagged_frames":[9,10,20]})
        self.assertFalse(report["automatically_approved"])
        self.assertTrue(original.exists())
        self.assertEqual(_sha(original),saved["original_mp4_sha256"])
        self.assertEqual(json.loads((self.output/
            "S02.needs_visual_review.json").read_text())["shot"],"S02")
        with self.assertRaisesRegex(CheckpointRejected,"overwrite"):
            checkpoint_raw(self.raw,self.receipt,self.reference,
                           self.output,"S02",17)

    def test_original_reference_mismatch_blocks_checkpoint(self):
        data=json.loads(self.receipt.read_text())
        data["reference_sha256"]="0"*64
        self.receipt.write_text(json.dumps(data))
        with self.assertRaisesRegex(CheckpointRejected,"mismatch"):
            checkpoint_raw(self.raw,self.receipt,self.reference,
                           self.output,"S02",17)
        self.assertFalse(self.output.joinpath("S02.original_wan_raw.mp4").exists())

    def test_original_source_checksum_mismatch_blocks_checkpoint(self):
        data=json.loads(self.receipt.read_text())
        data["clip_sha256"]="a"*64
        self.receipt.write_text(json.dumps(data))
        with self.assertRaisesRegex(CheckpointRejected,"mismatch"):
            checkpoint_raw(self.raw,self.receipt,self.reference,
                           self.output,"S02",17)
        self.assertFalse(self.output.joinpath("S02.original_wan_raw.mp4").exists())

    def test_historical_good_shots_cannot_be_overwritten(self):
        for sid in ("S00","S01"):
            with self.assertRaisesRegex(CheckpointRejected,"not be regenerated"):
                checkpoint_raw(self.raw,self.receipt,self.reference,
                               self.output,sid,17)

    def test_private_review_contact_has_frame_numbers_no_approvals(self):
        preview=self.folder/"S02.LOW_RES_FRAME_REVIEW_ONLY.jpg"
        result=selected_contact_sheet(self.raw,preview,[0,8,9,10,20])
        self.assertEqual(result["frames"],[0,8,9,10,20])
        with Image.open(preview) as img:
            self.assertEqual(img.size,(640,628))
        self.assertFalse(result["model_art_approved"])
        with self.assertRaisesRegex(CheckpointRejected,"overwrite"):
            selected_contact_sheet(self.raw,preview,[0,8])
        self.assertLess(preview.stat().st_size,3*1024*1024)


if __name__=="__main__":
    unittest.main()
