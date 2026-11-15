"""Cinematic 40+ second time-base QA without spending GPU or making fake frames."""
import copy
import unittest

from studio.eb01_v8_40s_native_motion_plan import (
    plan,verify_real_neural_sources,authorize_v8_scale_up,FPS,SEGMENT_NATIVE_FRAMES,
)
from studio.eb01_v7_storyboard import SHOTS,VOICE_SECONDS,VOICE_SHA


def all_fake_receipts_that_are_only_FOR_UNIT_TESTS():
    """Receipts are synthetic test data, never video files or release approval."""
    p=plan()
    result={}
    n=1
    for row in p["scenes"]:
        src=[]
        previous=None
        for idx in range(row["native_segment_count"]):
            first=previous if previous else f"{n+210000:064x}"
            last=f"{n+310000:064x}"
            item={
                "mp4_sha256":f"{n:064x}",
                "conditioning_input_frame_sha256":first,
                "last_decoded_frame_sha256":last,
                "native_frames":33,"native_fps":24,
                "native_width":576,"native_height":1024,
                "actual_model_inference":True,
                "canonical_actor_sha256":row["canonical_reference_sha256"],
                "rgb_neon_qc":"PASS",
                "identity_artistic_qc":"PASS",
            }
            if idx==0:
                item["initial_reference_sha256"]=row["canonical_reference_sha256"]
            src.append(item)
            previous=last
            n+=1
        result[row["scene_id"]]=src
    return result


class ElizabethNative40sTimelineTests(unittest.TestCase):
    def test_original_23_second_timeline_is_mathematically_invalid_for_target(self):
        original=sum(r[2] for r in SHOTS)
        self.assertEqual(original,563)
        self.assertAlmostEqual(original/FPS,23.4583333333,places=5)
        p=plan()
        self.assertEqual(p["original_v7_native_frames"],563)
        self.assertEqual(p["final_unique_native_frames"],967)
        self.assertGreater(p["final_duration_seconds"],40)
        self.assertEqual(p["new_true_inference_segment_count"],30)
        self.assertEqual(p["neural_frames_to_generate"],990)
        self.assertEqual(p["required_native_width"],576)
        self.assertEqual(p["required_native_height"],1024)
        self.assertFalse(p["production_executed"])
        self.assertFalse(p["cinematic_owner_approved"])

    def test_7_original_characters_dialogues_preserved_with_nonempty_acting_beats(self):
        p=plan()
        self.assertEqual(len(p["scenes"]),7)
        self.assertEqual(p["original_russian_qwen_lines"],6)
        self.assertEqual(p["original_character_actors"],2)
        self.assertEqual([r["scene_id"] for r in p["scenes"]],list(VOICE_SHA)[:0]+[x[0] for x in SHOTS])
        for row in p["scenes"]:
            self.assertEqual(row["unique_movie_frames"],
                             1+32*row["native_segment_count"])
            self.assertEqual(row["native_frames_per_segment"],SEGMENT_NATIVE_FRAMES)
            if row["scene_id"]!="S00":
                speak=round(VOICE_SECONDS[row["scene_id"]]*FPS)
                self.assertIsNotNone(row["canonical_voice_sha256"])
                self.assertGreaterEqual(row["dialogue_end_scene_frame"]-
                                        row["dialogue_start_scene_frame"],speak)
                self.assertGreater(row["dialogue_start_scene_frame"],0)
                self.assertLess(row["dialogue_end_scene_frame"],row["unique_movie_frames"])

    def test_full_synthetic_evidence_is_still_not_a_finished_film(self):
        evidence=all_fake_receipts_that_are_only_FOR_UNIT_TESTS()
        result=verify_real_neural_sources(evidence)
        self.assertEqual(result["status"],"SEGMENT_QA_PASS_OWNER_REVIEW_STILL_REQUIRED")
        self.assertEqual(result["genuinely_original_segments_verified"],30)
        self.assertTrue(result["native_576x1024_detail_verified"])
        self.assertFalse(result["full_40s_movie_ready"])

    def test_320x576_upscaled_pilot_cannot_satisfy_native_detail_requirement(self):
        evidence=all_fake_receipts_that_are_only_FOR_UNIT_TESTS()
        evidence["S02"][0]["native_width"]=320
        evidence["S02"][0]["native_height"]=576
        result=verify_real_neural_sources(evidence)
        self.assertEqual(result["status"],"BLOCKED")
        self.assertIn("S02_0_NATIVE_576X1024_DETAIL_NOT_PROVEN",result["reasons"])
        self.assertFalse(result["native_576x1024_detail_verified"])

    def test_boolean_or_string_dimensions_are_not_native_detail_proof(self):
        for width,height in ((True,1024),("576",1024),(576,False),(576,"1024")):
            evidence=all_fake_receipts_that_are_only_FOR_UNIT_TESTS()
            evidence["S04"][1]["native_width"]=width
            evidence["S04"][1]["native_height"]=height
            result=verify_real_neural_sources(evidence)
            self.assertEqual(result["status"],"BLOCKED")
            self.assertIn("S04_1_NATIVE_576X1024_DETAIL_NOT_PROVEN",result["reasons"])

    def test_duplicate_native_clip_is_rejected(self):
        evidence=all_fake_receipts_that_are_only_FOR_UNIT_TESTS()
        evidence["S01"][0]["mp4_sha256"]=evidence["S00"][0]["mp4_sha256"]
        result=verify_real_neural_sources(evidence)
        self.assertEqual(result["status"],"BLOCKED")
        self.assertIn("S01_0_REAL_NATIVE_GPU_OR_QA_NOT_PROVEN",result["reasons"])

    def test_missing_original_motion_segment_is_rejected(self):
        evidence=all_fake_receipts_that_are_only_FOR_UNIT_TESTS()
        evidence["S04"].pop()
        result=verify_real_neural_sources(evidence)
        self.assertIn("S04_MISSING_DISTINCT_NATIVE_I2V_SEGMENTS",result["reasons"])

    def test_jumping_to_unrelated_skull_frame_breaks_physical_continuity(self):
        evidence=all_fake_receipts_that_are_only_FOR_UNIT_TESTS()
        evidence["S02"][2]["conditioning_input_frame_sha256"]="f"*64
        result=verify_real_neural_sources(evidence)
        self.assertIn("S02_2_CONDITIONING_FRAME_CONTINUITY_BROKEN",result["reasons"])

    def test_unapproved_neon_or_fabricated_ai_never_counts(self):
        evidence=all_fake_receipts_that_are_only_FOR_UNIT_TESTS()
        evidence["S06"][0]["rgb_neon_qc"]="REVIEW"
        evidence["S06"][1]["actual_model_inference"]=False
        result=verify_real_neural_sources(evidence)
        self.assertEqual(result["status"],"BLOCKED")
        self.assertIn("S06_0_REAL_NATIVE_GPU_OR_QA_NOT_PROVEN",result["reasons"])
        self.assertIn("S06_1_REAL_NATIVE_GPU_OR_QA_NOT_PROVEN",result["reasons"])


    def test_malformed_or_overflowing_duration_is_blocked_without_exception(self):
        for duration in (True, "2.5", float("nan"), float("inf"), 10 ** 10000):
            pilot = valid_scaleup_pilot()
            pilot["pilot_duration_seconds"] = duration
            result = authorize_v8_scale_up(pilot, expected_pilot_sha256="a" * 64)
            self.assertFalse(result["scale_up_authorized"])
            self.assertIn("controlled_2_to_4_second_pilot_missing", result["blocking_gates"])

    def test_synthetic_or_unverified_inference_cannot_authorize_scaleup(self):
        pilot = valid_scaleup_pilot()
        pilot["actual_model_inference"] = False
        result = authorize_v8_scale_up(pilot, expected_pilot_sha256="a" * 64)
        self.assertFalse(result["scale_up_authorized"])
        self.assertIn("true_neural_inference_not_proven", result["blocking_gates"])


def valid_scaleup_pilot():
    sha = "a" * 64
    return {
        "pilot_mp4_sha256": sha,
        "owner_reviewed_mp4_sha256": sha,
        "decoded_frames": 60,
        "native_width": 576,
        "native_height": 1024,
        "native_fps": 24,
        "pilot_duration_seconds": 2.5,
        "actual_model_inference": True,
        "rgb_status": "NO_NEON_ARTIFACT_FOUND",
        "owner_visual_review": "PASS",
        "skeletal_anatomy_review": "PASS",
        "human_identity_review": "PASS",
        "acting_motion_review": "PASS",
        "private_media_acl_verified": True,
        "audio_conditioned_lipsync_review": "PASS",
    }


class V8ScaleUpAuthorizationTests(unittest.TestCase):
    def test_exact_owner_reviewed_2_to_4s_native_pilot_can_authorize_planning_only(self):
        pilot = valid_scaleup_pilot()
        result = authorize_v8_scale_up(pilot, expected_pilot_sha256="a" * 64)
        self.assertTrue(result["scale_up_authorized"])
        self.assertEqual(result["blocking_gates"], [])
        self.assertFalse(result["gpu_job_submitted"])
        self.assertFalse(result["full_movie_ready"])
        self.assertFalse(result["public_media_export"])

    def test_current_33f_320x576_a609_pilot_cannot_authorize_batch(self):
        pilot = {
            "pilot_mp4_sha256": "1f13fdbc6b9bc3dc42c3bdab75090ac025556b0c6d63b4ff572769320fb7d71f",
            "decoded_frames": 33,
            "native_width": 320,
            "native_height": 576,
            "native_fps": 24,
            "pilot_duration_seconds": 1.375,
            "rgb_status": "NO_NEON_ARTIFACT_FOUND",
        }
        result = authorize_v8_scale_up(
            pilot,
            expected_pilot_sha256="1f13fdbc6b9bc3dc42c3bdab75090ac025556b0c6d63b4ff572769320fb7d71f",
        )
        self.assertFalse(result["scale_up_authorized"])
        self.assertIn("controlled_2_to_4_second_pilot_missing", result["blocking_gates"])
        self.assertIn("native_576x1024_detail_missing", result["blocking_gates"])
        self.assertIn("private_owner_visual_signoff", result["blocking_gates"])

    def test_stale_or_unreviewed_sha_is_rejected(self):
        pilot = valid_scaleup_pilot()
        pilot["owner_reviewed_mp4_sha256"] = "b" * 64
        result = authorize_v8_scale_up(pilot, expected_pilot_sha256="a" * 64)
        self.assertFalse(result["scale_up_authorized"])
        self.assertIn("authentic_pilot_sha_not_owner_reviewed", result["blocking_gates"])

    def test_missing_expected_sha_never_authorizes_scale_up(self):
        result = authorize_v8_scale_up(valid_scaleup_pilot())
        self.assertFalse(result["scale_up_authorized"])
        self.assertIn("expected_pilot_sha_not_pinned", result["blocking_gates"])

    def test_boolean_and_string_dimensions_are_not_native_detail(self):
        for width, height in ((True, 1024), ("576", 1024), (576, False), (576, "1024")):
            pilot = valid_scaleup_pilot()
            pilot["native_width"] = width
            pilot["native_height"] = height
            result = authorize_v8_scale_up(pilot, expected_pilot_sha256="a" * 64)
            self.assertFalse(result["scale_up_authorized"])
            self.assertIn("native_576x1024_detail_missing", result["blocking_gates"])

    def test_duration_must_match_24fps_decoded_frame_count(self):
        pilot = valid_scaleup_pilot()
        pilot["pilot_duration_seconds"] = 3.0
        result = authorize_v8_scale_up(pilot, expected_pilot_sha256="a" * 64)
        self.assertFalse(result["scale_up_authorized"])
        self.assertIn("controlled_2_to_4_second_pilot_missing", result["blocking_gates"])

if __name__ == "__main__":
    unittest.main()
