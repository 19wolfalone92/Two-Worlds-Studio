"""Studio recliner-native LTX contracts: no Freevideo, no fake final video."""
import ast
import hashlib
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tools/kaggle_studio_recliner_native.py"
USER_IMAGE_SHA = "b2a235c7493df7d7d2cb98572d5baa27836511572172abb4a911784ae93a2e21"

class ReclinerNativeStudioTests(unittest.TestCase):
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
