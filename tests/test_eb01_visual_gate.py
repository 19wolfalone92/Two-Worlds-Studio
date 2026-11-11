"""CPU-only regression checks: no synthetic fixture may bypass EB01 visual sign-off."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from PIL import Image

from tools.eb01_visual_gate import CAST, CHECKS, ReviewBlocked, audit, sha256


class EB01VisualGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.review = self.root / "review.json"
        def picture(name):
            path = self.root / (name + ".png")
            pic = Image.effect_noise((576, 1024), 45).convert("RGB")
            pic.save(path, format="PNG")
            return {"file": path.name, "sha256": sha256(path)}
        self.data = {
            "schema": 1, "episode": "EB01", "approved": True,
            "human_visual_review": True, "art_origin": "real_images",
            "references": {
                who: {**picture(who + "_canon"), "approved": True}
                for who in ("elizabeth", "kostya")
            },
            "scenes": [
                {"id": scene, **picture("frame_" + scene),
                 "characters": sorted(CAST[scene]),
                 "visual_checks": {key: True for key in CHECKS}}
                for scene in ("S00", "S01", "S02", "S03", "S04", "S05", "S06")
            ]
        }
        self.save()

    def save(self):
        self.review.write_text(json.dumps(self.data), encoding="utf-8")

    def test_valid_reviewed_storyboard_passes_without_gpu(self):
        result = audit(self.root, self.review)
        self.assertEqual(result["status"], "VISUAL_REVIEW_APPROVED")
        self.assertEqual(result["gpu_jobs_launched"], 0)
        self.assertFalse(result["published"])
        self.assertFalse(result["motion_approved"])

    def test_human_instead_of_skeleton_fails_even_when_mp4_would_be_decodable(self):
        self.data["scenes"][2]["visual_checks"]["no_human_instead_of_skeleton"] = False
        self.save()
        with self.assertRaisesRegex(ReviewBlocked, "quality/continuity"):
            audit(self.root, self.review)

    def test_unreviewed_references_and_missing_approval_block(self):
        self.data["references"]["kostya"]["approved"] = False
        self.save()
        with self.assertRaisesRegex(ReviewBlocked, "Character canon"):
            audit(self.root, self.review)
        self.data["references"]["kostya"]["approved"] = True
        self.data["approved"] = False
        self.save()
        with self.assertRaisesRegex(ReviewBlocked, "not personally accepted"):
            audit(self.root, self.review)

    def test_changed_visuals_block_even_after_review(self):
        p = self.root / self.data["scenes"][4]["file"]
        p.write_bytes(p.read_bytes() + b"modified")
        with self.assertRaisesRegex(ReviewBlocked, "Frame changed"):
            audit(self.root, self.review)

    def test_no_real_motion_pilot_means_no_full_video(self):
        with self.assertRaisesRegex(ReviewBlocked, "motion"):
            audit(self.root, self.review, require_motion=True)

    def test_false_camera_zoom_is_not_character_animation(self):
        clip = self.root / "motion.mp4"
        clip.write_bytes(b"placeholder")
        self.data["motion_pilot"] = {
            "file": "motion.mp4", "sha256": sha256(clip),
            "human_approved": True, "kind": "camera_pan",
            "neural_actor_motion": False}
        self.save()
        with self.assertRaisesRegex(ReviewBlocked, "Camera pan"):
            audit(self.root, self.review, require_motion=True)

    def test_real_video_requires_approved_motion_character_details(self):
        clip = self.root / "motion.mp4"
        clip.write_bytes(b"not-a-valid-mp4-test-only")
        self.data["motion_pilot"] = {
            "file": "motion.mp4", "sha256": sha256(clip),
            "human_approved": True, "kind": "image_to_video",
            "neural_actor_motion": True,
            "face_stable": True, "skeleton_stable": False,
            "physical_movement": True, "natural_motion": True}
        self.save()
        fake_probe=SimpleNamespace(stdout=json.dumps({
            "format": {"duration": "2.5"},
            "streams": [{"codec_type": "video"}]}))
        with patch("tools.eb01_visual_gate.subprocess.run", return_value=fake_probe):
            with self.assertRaisesRegex(ReviewBlocked, "skeleton_stable"):
                audit(self.root, self.review, require_motion=True)
        self.data["motion_pilot"]["skeleton_stable"] = True
        self.save()
        with patch("tools.eb01_visual_gate.subprocess.run", return_value=fake_probe):
            result=audit(self.root, self.review, require_motion=True)
            self.assertTrue(result["motion_approved"])


if __name__ == "__main__":
    unittest.main()
