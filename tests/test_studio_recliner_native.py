"""Studio recliner-native LTX contracts: no Freevideo, no fake final video."""
import ast
import hashlib
import unittest

from tools.kaggle_studio_recliner_native import _validate_recliner_video_stream, _classify_recliner_color_qa
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tools/kaggle_studio_recliner_native.py"
USER_IMAGE_SHA = "9b1310c91f9718e29d026d6346dbc3ebd7687027a871ac5dbdf4a4f2e36895b3"

class ReclinerNativeStudioTests(unittest.TestCase):

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

    def test_no_static_photo_video_substitutions(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("original_neural_segments",source)
        self.assertIn("image_cond_noise_scale=0.15",source)
        self.assertIn("conditioning_media_paths=",source)
        self.assertIn('"extra_interpolated_frames":0',source)
        self.assertIn('human_artistic_approved":False',source)
        self.assertNotIn("loop=1",source)
        self.assertNotIn("minterpolate",source)

if __name__ == "__main__":
    unittest.main()
