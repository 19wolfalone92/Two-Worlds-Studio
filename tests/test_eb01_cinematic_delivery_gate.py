"""Regression suite: green CI must not imply an acceptable EB01 movie."""
import copy
import unittest

from studio.eb01_cinematic_delivery_gate import assess, SCENES, DIALOGUE

def good():
    report={
       "sha256":"0"*64,
       "duration_seconds":44.5,"resolution":[720,1280],"fps":24,
       "scenes_on_timeline":7,"russian_voice_lines":6,
       "russian_subtitle_cues":6,
       "full_visual_frame_qa_status":"PASS",
       "suspected_colored_neon_frames":0,
       "original_artistic_quality_approved":None,
    }
    scenes={}
    for idx,sid in enumerate(SCENES):
        scenes[sid]={
            "actor_id":("ELIZABETH" if sid in ("S01","S03","S05") else "KOSTYA"),
            "skeleton_anatomy_preserved":("N/A" if sid in ("S01","S03","S05") else "PASS"),
            "source_sha256":f"{idx+1:064x}",
            "model_receipt_sha256":f"{idx+8:064x}",
            "canonical_character_reference_sha256":f"{idx+16:064x}",
            "original_native_ai_inference":True,
            "real_neural_video_verified":True,
            "same_character_identity_qc":"PASS",
            "physically_coherent_movement_qc":"PASS",
            "human_visual_frame_review":"PASS",
            "neon_color_qc":"PASS",
        }
        if sid in DIALOGUE:
            scenes[sid].update({
                "original_qwen_voice_sha256":f"{idx+24:064x}",
                "audio_conditioned_lipsync":"PASS",
                "mouth_jaw_cheek_motion_review":"PASS",
                "bone_jaw_hinge_motion_review":"PASS",
            })
    return report,scenes


class CinematicReleaseGateTests(unittest.TestCase):
    def test_actual_rejected_23_second_movie_is_blocked(self):
        report,scenes=good()
        report.update({
           "sha256":"5bc0180346a9e89c80e72cfea2ef4ca81bbbd76186c0bc62af1634b29f2b3b1f",
           "duration_seconds":23.45833,"resolution":[320,576],
           "suspected_colored_neon_frames":76,
           "full_visual_frame_qa_status":"REVIEW",
           "original_artistic_quality_approved":False})
        scenes["S02"]["original_native_ai_inference"]=False
        scenes["S02"]["reused_frames_from_S00_S04_S06"]=True
        result=assess(report,evidence=scenes)
        self.assertEqual(result["status"],"BLOCKED")
        self.assertIn("DURATION_SHORTER_THAN_40_SECONDS",result["reasons"])
        self.assertIn("INSUFFICIENT_NATIVE_VERTICAL_RESOLUTION",result["reasons"])
        self.assertIn("NEON_COLOR_SPECTRAL_ARTIFACTS",result["reasons"])
        self.assertIn("S02_FOOTAGE_REUSE_OR_NO_ORIGINAL_INFERENCE",result["reasons"])
        self.assertFalse(result["cinematic_delivery_approved"])

    def test_full_good_video_remains_only_owner_review_without_owner_approval(self):
        r,scenes=good()
        result=assess(r,evidence=scenes)
        self.assertEqual(result["status"],"OWNER_REVIEW_REQUIRED")
        self.assertFalse(result["cinematic_delivery_approved"])
        self.assertEqual(result["reasons"],[])

    def test_owner_approval_is_separate_from_automatic_model_receipt(self):
        r,scenes=good()
        self.assertEqual(assess(r,evidence=scenes,owner_artistic_approval=True)["status"],
                         "RELEASE_REVIEW_PASS")
        r["original_artistic_quality_approved"]=False
        self.assertEqual(assess(r,evidence=scenes,owner_artistic_approval=True)["status"],
                         "BLOCKED")

    def test_missing_full_film_sha_cannot_be_approved_even_when_actor_receipts_good(self):
        r,scenes=good()
        del r["sha256"]
        self.assertIn("REAL_FILM_FILE_SHA256_NOT_PROVEN",
                      assess(r,evidence=scenes,owner_artistic_approval=True)["reasons"])

    def test_missing_or_wrong_type_rgb_count_never_means_zero_bad_frames(self):
        for value in (None,False,"0",-1):
            r,scenes=good()
            if value is None:
                del r["suspected_colored_neon_frames"]
            else:
                r["suspected_colored_neon_frames"]=value
            self.assertIn("NEON_COLOR_SPECTRAL_ARTIFACTS",
                          assess(r,evidence=scenes)["reasons"])

    def test_no_lipsync_cannot_be_green_video(self):
        r,scenes=good()
        scenes["S05"]["audio_conditioned_lipsync"]="UNKNOWN"
        self.assertIn("S05_NO_VERIFIED_AUDIO_DRIVEN_LIPSYNC",
                      assess(r,evidence=scenes)["reasons"])

    def test_repeated_actor_footage_cannot_masquerade_as_seven_unique_performances(self):
        r,scenes=good()
        scenes["S02"]["source_sha256"]=scenes["S04"]["source_sha256"]
        result=assess(r,evidence=scenes)
        self.assertIn("S04_DUPLICATE_SOURCE_CLIP",result["reasons"])

    def test_missing_receipts_for_any_scene_is_blocked(self):
        r,scenes=good()
        scenes["S06"].pop("model_receipt_sha256")
        self.assertIn("S06_ORIGINAL_VIDEO_IDENTITY_MOTION_UNVERIFIED",
                      assess(r,evidence=scenes)["reasons"])

    def test_missing_scene_cannot_be_silently_approved(self):
        r,scenes=good()
        del scenes["S02"]
        result=assess(r,evidence=scenes)
        self.assertIn("MISSING_INDEPENDENT_SCENE_RECEIPTS",result["reasons"])
        self.assertIn("S02_NO_VERIFIED_AUDIO_DRIVEN_LIPSYNC",result["reasons"])

    def test_no_self_certification_from_a_successful_ffmpeg_status(self):
        r,scenes=good()
        r["quality_check"]="PASS"
        r["ffmpeg_export"]="SUCCESS"
        del scenes["S01"]["human_visual_frame_review"]
        self.assertIn("S01_ORIGINAL_VIDEO_IDENTITY_MOTION_UNVERIFIED",
                      assess(r,evidence=scenes)["reasons"])

    def test_only_review_mode_when_no_human_signoff(self):
        r,scenes=good()
        r["original_artistic_quality_approved"]=False
        self.assertEqual(assess(r,evidence=scenes)["status"],"OWNER_REVIEW_REQUIRED")

    def test_reject_no_real_provenance_even_with_owner_approval(self):
        r,scenes=good()
        scenes["S03"]["model_receipt_sha256"]="not-real"
        self.assertEqual(assess(r,evidence=scenes,owner_artistic_approval=True)["status"],
                         "BLOCKED")

    def test_skeleton_cannot_pass_on_human_mouth_only(self):
        r,scenes=good()
        scenes["S02"]["bone_jaw_hinge_motion_review"]="UNKNOWN"
        scenes["S02"]["mouth_jaw_cheek_motion_review"]="PASS"
        self.assertIn("S02_NO_VERIFIED_AUDIO_DRIVEN_LIPSYNC",
                      assess(r,evidence=scenes)["reasons"])

    def test_skeleton_cannot_grow_human_flesh_or_lose_suit(self):
        r,scenes=good()
        scenes["S06"]["skeleton_anatomy_preserved"]="FAILED"
        self.assertIn("S06_BONE_ANATOMY_OR_SUIT_DRIFT_UNVERIFIED",
                      assess(r,evidence=scenes)["reasons"])

    def test_elizabeth_cannot_pass_on_skeleton_jaw_only(self):
        r,scenes=good()
        scenes["S03"]["mouth_jaw_cheek_motion_review"]="UNKNOWN"
        scenes["S03"]["bone_jaw_hinge_motion_review"]="PASS"
        self.assertIn("S03_NO_VERIFIED_AUDIO_DRIVEN_LIPSYNC",
                      assess(r,evidence=scenes)["reasons"])

    def test_skeletal_actor_cannot_be_replaced_by_different_character(self):
        r,scenes=good()
        scenes["S04"]["actor_id"]="ELIZABETH"
        self.assertIn("S04_ACTOR_IDENTITY_PROVENANCE_MISSING",
                      assess(r,evidence=scenes)["reasons"])

if __name__=="__main__":
    unittest.main()
