"""Cinematic 40+ second time-base QA without spending GPU or making fake frames."""
import copy
import unittest

from studio.eb01_v8_40s_native_motion_plan import (
    plan,verify_real_neural_sources,FPS,SEGMENT_NATIVE_FRAMES,
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
                "first_frame_sha256":first,
                "last_frame_sha256":last,
                "native_frames":33,"native_fps":24,
                "actual_model_inference":True,
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
        self.assertFalse(result["full_40s_movie_ready"])

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
        evidence["S02"][2]["first_frame_sha256"]="f"*64
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


if __name__=="__main__":
    unittest.main()
