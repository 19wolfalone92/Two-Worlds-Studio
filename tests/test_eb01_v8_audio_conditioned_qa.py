"""Regression tests for V8 original-voice performance evidence gate."""
import unittest
from studio.eb01_v8_audio_conditioned_qa import verify_audio_conditioned_performances
from studio.eb01_v8_40s_native_motion_plan import plan


def evidence():
    return {
        s["scene_id"]: {
            "scene_id": s["scene_id"],
            "voice_sha256": s["canonical_voice_sha256"],
            "dialogue_start_frame": s["dialogue_start_scene_frame"],
            "dialogue_end_frame": s["dialogue_end_scene_frame"],
            "decoded_scene_frames": s["unique_movie_frames"],
            "original_neural_frames": True,
            "audio_visual_offset_ms": 12,
            "jaw_audio_correlation": 0.7,
            "voiced_jaw_motion_px": 4,
            "silent_jaw_motion_px": 0.2,
            "anatomical_artistic_owner_review": "PASS",
        } for s in plan()["scenes"] if s["canonical_voice_sha256"]
    }


class AudioConditionedQATests(unittest.TestCase):
    def test_valid_metadata_not_final_film(self):
        result = verify_audio_conditioned_performances(evidence())
        self.assertEqual(result["verified_lines"], 6)
        self.assertFalse(result["movie_ready"])

    def test_missing_line(self):
        e = evidence()
        e.pop("S02")
        self.assertEqual(verify_audio_conditioned_performances(e)["status"], "BLOCKED")

    def test_frozen_jaw_rejected(self):
        e = evidence()
        e["S02"]["voiced_jaw_motion_px"] = 0.1
        self.assertEqual(verify_audio_conditioned_performances(e)["status"], "BLOCKED")

    def test_nonfinite_correlation_rejected(self):
        e = evidence()
        e["S02"]["jaw_audio_correlation"] = float("nan")
        self.assertEqual(verify_audio_conditioned_performances(e)["status"], "BLOCKED")

    def test_voice_sha_mismatch_rejected(self):
        e = evidence()
        e["S02"]["voice_sha256"] = "0" * 64
        self.assertEqual(verify_audio_conditioned_performances(e)["status"], "BLOCKED")

    def test_owner_anatomy_review_required(self):
        e = evidence()
        e["S02"]["anatomical_artistic_owner_review"] = "PENDING"
        self.assertEqual(verify_audio_conditioned_performances(e)["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
