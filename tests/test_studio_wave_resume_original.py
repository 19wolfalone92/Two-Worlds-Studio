"""Offline regression tests for real Studio-wave Kaggle MP4 recovery."""
from __future__ import annotations
import base64
import json
import tempfile
import unittest
from pathlib import Path

from tools.studio_wave_resume_original import (
    SEGMENT_SHA, stage_elapsed_seconds, private_source_reference,
)
from tools.studio_wave_status_gate import SOURCE_SHA256

class RecoveryTests(unittest.TestCase):
    def test_event_times_are_read_from_original_kaggle_log_not_invented(self):
        rows=[
            {"stream_name":"stdout","time":13.5,
             "data":"STUDIO_WAVE_NATIVE_LTX_SAMPLE_START\n"},
            {"stream_name":"stdout","time":355.75,
             "data":"WAVE_STAGE_NATIVE_VIDEO_OUTPUT_READY\n"},
        ]
        stream="[\n"+json.dumps(rows[0])+"\n,"+json.dumps(rows[1])+"\n]"
        self.assertEqual(stage_elapsed_seconds(stream),342.25)
        self.assertEqual(stage_elapsed_seconds(json.dumps(rows)),342.25)
    def test_no_timing_without_begin_and_end_stage_attestation(self):
        rows=[{"time":18.0,"data":"WAVE_STAGE_END_DECODE"}]
        with self.assertRaisesRegex(RuntimeError,"TIMING_NOT_PROVEN"):
            stage_elapsed_seconds(json.dumps(rows))
    def test_missing_owner_photo_is_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError,"AMBIGUOUS"):
            private_source_reference('IMAGE_B64="ZGF0YQ=="')
    def test_private_model_template_contains_no_real_source_media(self):
        source=Path("tools/kaggle_studio_man_wave_native.py").read_text()
        self.assertEqual(source.count('__STUDIO_WAVE_IMAGE_B64__'),1)
        self.assertEqual(source.count('__STUDIO_WAVE_CHECKPOINT_B64__'),2)
        self.assertIn('stage_prefix="WAVE"',source)
        self.assertIn('recovered_from_prior_private_kernel',source)
        self.assertEqual(len(SEGMENT_SHA),64)
        self.assertEqual(len(SOURCE_SHA256),64)
    def test_preparation_never_submits_or_publishes_media(self):
        source=Path("tools/studio_wave_resume_original.py").read_text()
        self.assertNotIn('kernels", "push"',source)
        self.assertNotIn("STUDIO_WAVE_TRUE_NATIVE_65F_24FPS.mp4",source)
        self.assertIn('STUDIO_WAVE_ORIGINAL_NEURAL_SEGMENT0.twenc',source)

if __name__ == "__main__":
    unittest.main()
