"""Studio recliner-native LTX contracts: no Freevideo, no fake final video."""
import ast
import hashlib
import unittest

from tools.kaggle_studio_recliner_native import _validate_recliner_video_stream, _classify_recliner_color_qa, _continuation_attempts, _recliner_region_motion_pass
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tools/kaggle_studio_recliner_native.py"
USER_IMAGE_SHA = "9b1310c91f9718e29d026d6346dbc3ebd7687027a871ac5dbdf4a4f2e36895b3"

class ReclinerNativeStudioTests(unittest.TestCase):

    def test_recliner_color_qa_blocks_inconsistent_clean_status_with_structural_flags(self):
        with self.assertRaises(RuntimeError):
            _classify_recliner_color_qa({
                "status": "NO_NEON_ARTIFACT_FOUND",
                "horizontal_neon_band_flagged_frames": [],
                "spectral_intrusion_flagged_frames": [],
                "neon_line_geometry_flagged_frames": [9],
            })


    def test_cool_lounge_palette_is_review_not_false_green(self):
        result = _classify_recliner_color_qa({
            "status": "REVIEW_PERSISTENT_COOL_CHROMA_NO_STRUCTURAL_PROOF",
            "horizontal_neon_band_flagged_frames": [],
            "spectral_intrusion_flagged_frames": [],
            "neon_line_geometry_flagged_frames": [],
        })
        self.assertTrue(result["review_required"])
        self.assertEqual(result["structural_flags"]["neon_line_geometry"], 0)

    def test_recliner_color_qa_still_blocks_structural_rgb_streaks(self):
        with self.assertRaises(RuntimeError):
            _classify_recliner_color_qa({
                "status": "REVIEW_PERSISTENT_COOL_CHROMA_NO_STRUCTURAL_PROOF",
                "horizontal_neon_band_flagged_frames": [],
                "spectral_intrusion_flagged_frames": [],
                "neon_line_geometry_flagged_frames": [4, 5],
            })


    def test_real_checkpoint_actor_motion_and_wall_only_failure(self):
        # Measured independently from the authenticated original and two native outputs.
        self.assertTrue(_recliner_region_motion_pass({"head":1.134,"torso":1.567,"wall":0.522},0))
        self.assertFalse(_recliner_region_motion_pass({"head":0.201,"torso":0.118,"wall":1.416},1))
        self.assertFalse(_recliner_region_motion_pass({"head":0.01,"torso":0.02,"wall":0.01},2))
        self.assertFalse(_recliner_region_motion_pass({"head":float("nan"),"torso":1.0,"wall":0.1},0))
        self.assertFalse(_recliner_region_motion_pass({"head":0.1,"torso":1.0,"wall":0.1},1))
        self.assertTrue(_recliner_region_motion_pass({"head":0.7,"torso":0.4,"wall":0.3},1))

    def test_recliner_native_probe_accepts_locked_three_by_four_frame_clock(self):
        result = _validate_recliner_video_stream({
            "codec_name": "h264", "width": "576", "height": "768", "avg_frame_rate": "24/1",
            "nb_read_frames": "33", "duration": "1.375000",
        })
        self.assertEqual((result["width"], result["height"], result["fps"], result["frames"]),
                         (576, 768, "24", 33))

    def test_recliner_native_probe_rejects_generic_vertical_nine_by_sixteen(self):
        with self.assertRaises(RuntimeError):
            _validate_recliner_video_stream({
                "codec_name": "h264", "width": "576", "height": "1024", "avg_frame_rate": "24/1",
                "nb_read_frames": "33", "duration": "1.375000",
            })

    def test_source_compiles_without_coupling_to_freevideo(self):
        source = SOURCE.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertNotIn("freevideo",source.lower())
        self.assertNotIn("runway",source.lower())
        self.assertNotIn("kling",source.lower())
        self.assertIn("Lightricks/LTX-Video",source)

    def test_owner_source_and_frame_origin_are_strictly_anchored(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn(USER_IMAGE_SHA,source)
        self.assertIn("photo.size!=(576,768)",source)
        self.assertIn("height=768,width=576,num_frames=33",source)
        self.assertIn("stage_prefix=\"WAVE\"",source)
        self.assertIn("run_native_bounded",source)
        self.assertIn("STUDIO_RECLINER_TRUE_NATIVE_129F_24FPS.mp4",source)
        self.assertIn('m.get("frames")!=129',source)
        self.assertIn('"output_frames":129',source)
        self.assertIn('if len(videos)!=4',source)
        self.assertIn('concat=n=4:v=1:a=0[out]',source)



    def test_continuation_uses_higher_noise_for_visible_motion_and_keeps_motion_gate(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("LTX_IMAGE_COND_NOISE_SCALE", source)
        self.assertEqual(_continuation_attempts(1), ((0.25,2026112302),(0.28,2026113311)))
        self.assertEqual(len(_continuation_attempts(0)),1)
        self.assertEqual(len(_continuation_attempts(3)),2)
        with self.assertRaises(ValueError):
            _continuation_attempts(4)
        self.assertIn('if rejection is not None:',source)
        self.assertLess(source.index('if rejection is not None:'),source.index('shutil.copy2(clip,safe)'))
        self.assertIn('_ACCEPTED.json',source)
        self.assertIn('"successful_native_segments":len(records)',source)
        self.assertIn("New native neural segment failed independent motion QA", source)
        self.assertIn('mean_frame_luma_delta"]<=0.18', source)

    def test_private_gpu_workflow_recovers_checkpoint_before_any_new_gpu_submission(self):
        workflow = (SOURCE.parents[1] / ".github" / "workflows" /
                    "studio-recliner-native-private-gpu.yml").read_text(encoding="utf-8")
        self.assertLess(workflow.index("CPU validate studio native LTX source"),
                        workflow.index("Recover and qualify original real segment 0"))
        self.assertLess(workflow.index("Recover and qualify original real segment 0"),
                        workflow.index("Resume from recovered segment 0 and generate only segments 1–3"))
        self.assertIn("STUDIO_PREVIOUS_KERNEL", workflow)
        self.assertIn("__STUDIO_RECLINER_CHECKPOINT_B64__", workflow)
        self.assertIn("STUDIO_RECLINER_PRIVATE_GPU_0.log", workflow)
        self.assertIn("qa_color_no_structural_artifacts", workflow)
        self.assertIn("if: failure()", workflow)
        self.assertIn("studio-recliner-native-encrypted-recovery", workflow)

    def test_no_static_photo_video_substitutions(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("original_neural_segments",source)
        self.assertIn("LTX_IMAGE_COND_NOISE_SCALE",source)
        self.assertIn("conditioning_media_paths=",source)
        self.assertIn('"extra_interpolated_frames":0',source)
        self.assertIn('human_artistic_approved":False',source)
        self.assertNotIn("loop=1",source)
        self.assertNotIn("minterpolate",source)

if __name__ == "__main__":
    unittest.main()
