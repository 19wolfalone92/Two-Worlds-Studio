"""Offline V6 tests: structural/video gate failures, never fake a completed film."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from studio.v6_production import (
    CAST, LINE_LOCK, MOTION, SCENE_CAST, SCENES, VISUAL,
    ProductionBlocked, checked_visuals, digest, pilot, plan, read_storyboard, render
)


class V6IdentityFirstTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "canon").mkdir()
        (self.root / "frames").mkdir()
        (self.root / "voice").mkdir()
        self.project = self.root / "storyboard.json"
        self.review = self.root / "visual_review.json"
        self.takes = self.root / "takes.json"

        def still(name):
            target = self.root / name
            # These are ENGINEERING TEST ASSETS, not publishable story images.
            image = Image.effect_noise((576, 1024), 30).convert("RGB")
            image.save(target)
            return digest(target)

        refs = {who: {"image": "canon/" + who + ".png",
                      "sha256": still("canon/" + who + ".png")} for who in CAST}
        shots = []
        for sid in SCENES:
            shot = {
                "id": sid, "characters": sorted(SCENE_CAST[sid]),
                "image": "frames/" + sid + ".png",
                "image_sha256": still("frames/" + sid + ".png"),
                "duration": 3.5, "caption": LINE_LOCK.get(sid, ""),
                "motion_prompt": "Natural carefully directed upper body and facial performance, with locked identity and believable physical movement.",
            }
            if sid != "S00":
                wav = self.root / "voice" / (sid + ".wav")
                wav.write_bytes(b"fake-wave-container-contract-only" * 200)
                shot["voice"] = "voice/" + sid + ".wav"
                shot["voice_sha256"] = digest(wav)
            shots.append(shot)
        self.story = {
            "schema": 6, "episode": "ELIZABETH_BONES_E01",
            "model": "wan22-5b", "fps": 24,
            "references": refs, "shots": shots,
        }
        self.write_story()
        self.review_data = {
            "schema": 6, "episode": "ELIZABETH_BONES_E01",
            "storyboard_sha256": digest(self.project),
            "human_visual_review": True, "approved": True,
            "reviewer": "owner", "reference_approved": dict.fromkeys(CAST, True),
            "visual_checks": {sid: dict.fromkeys(VISUAL, True) for sid in SCENES},
        }
        self.write_review()

    def write_story(self):
        self.project.write_text(json.dumps(self.story, ensure_ascii=False), encoding="utf-8")

    def write_review(self):
        self.review.write_text(json.dumps(self.review_data, ensure_ascii=False), encoding="utf-8")

    def test_plan_records_no_gpu_and_full_six_lines(self):
        x = plan(self.project)
        self.assertEqual(x["status"], "PLAN_ONLY_NO_GPU")
        self.assertEqual(x["spoken_lines"], 6)
        self.assertFalse(x["neural_inference_performed"])

    def test_lock_all_source_images_for_identity(self):
        story, review, frames = checked_visuals(self.root, self.project, self.review)
        self.assertEqual(len(frames), 7)
        self.assertEqual(story["episode"], "ELIZABETH_BONES_E01")

    def test_kostya_flesh_human_has_to_be_rejected(self):
        self.review_data["visual_checks"]["S02"]["kostya_is_skeleton"] = False
        self.write_review()
        with self.assertRaisesRegex(ProductionBlocked, "anatomy"):
            checked_visuals(self.root, self.project, self.review)

    def test_frame_changes_after_approval_blocked(self):
        target = self.root / "frames" / "S06.png"
        target.write_bytes(target.read_bytes() + b"edited-after-approval")
        with self.assertRaisesRegex(ProductionBlocked, "Image changed"):
            checked_visuals(self.root, self.project, self.review)

    def test_reviewer_not_auto_invented(self):
        self.review_data["reviewer"] = "automatic_model"
        self.write_review()
        with self.assertRaisesRegex(ProductionBlocked, "human-reviewed"):
            checked_visuals(self.root, self.project, self.review)

    def test_wrong_locked_dialogue_is_blocked(self):
        self.story["shots"][4]["caption"] = "Arbitrary invented dialogue"
        self.write_story()
        with self.assertRaisesRegex(ProductionBlocked, "six-line"):
            read_storyboard(self.project)

    def test_unapproved_gpu_and_dry_run(self):
        x = pilot(self.project, self.review, self.root, "S02",
                  execute=False, accept_license=False, host="http://127.0.0.1:8188")
        self.assertEqual(x["status"], "PILOT_READY_NO_GPU")
        self.assertFalse(x["gpu_executed"])
        with self.assertRaisesRegex(ProductionBlocked, "licence"):
            pilot(self.project, self.review, self.root, "S02",
                  execute=True, accept_license=False, host="http://127.0.0.1:8188")

    def test_render_no_takes_no_full_video(self):
        self.takes.write_text(json.dumps({
            "schema": 6, "storyboard_sha256": digest(self.project),
            "shots": {}}), encoding="utf-8")
        with self.assertRaisesRegex(ProductionBlocked, "Seven full"):
            render(self.project, self.review, self.takes, self.root, "review.mp4")
        self.assertFalse((self.root / "renders" / "review.mp4").exists())

    def test_render_no_approval_without_true_clip(self):
        takes = {"schema": 6, "storyboard_sha256": digest(self.project),
                 "shots": {sid: {} for sid in SCENES}}
        self.takes.write_text(json.dumps(takes), encoding="utf-8")
        with self.assertRaisesRegex(ProductionBlocked, "Human-approved motion pilot"):
            render(self.project, self.review, self.takes, self.root, "review.mp4")

    def test_render_hands_native_editor_only_real_clips(self):
        (self.root / "clips").mkdir()
        takes = {"schema": 6, "storyboard_sha256": digest(self.project), "shots": {}}
        reviews = {}
        for sid in SCENES:
            clip = self.root / "clips" / (sid + ".mp4")
            clip.write_bytes(b"mock-generated-video-for-orchestration-only")
            receipt = clip.with_suffix(".receipt.json")
            receipt.write_text("{}", encoding="utf-8")
            takes["shots"][sid] = {
                "clip": "clips/" + sid + ".mp4",
                "receipt": "clips/" + sid + ".receipt.json", "sha256": digest(clip),
            }
            reviews[sid] = {
                "sha256": digest(clip), "approved": True, "reviewer": "owner",
                **dict.fromkeys(MOTION, True),
            }
        self.takes.write_text(json.dumps(takes), encoding="utf-8")
        self.review_data.update(pilot_approved=True, motion_checks=reviews)
        self.write_review()
        calls = []

        def verified_clip(root, sid, row, frame, duration):
            return self.root / row["clip"], {"inference_performed": True}

        def fake_native(manifest, output, **options):
            obj = json.loads(manifest.read_text(encoding="utf-8"))
            calls.append(obj)
            self.assertEqual(len(obj["shots"]), 7)
            self.assertTrue(all("clip" in x and "image" not in x for x in obj["shots"]))
            self.assertTrue(all(x["clip_take"]["engine"] == "neural_comfy" for x in obj["shots"]))
            self.assertEqual(sum(bool(x.get("voice")) for x in obj["shots"]), 6)
            self.assertFalse(options.get("master", True))
            output.write_bytes(b"fake-final-orchestration-only-not-a-video")
            return {"shots": 7, "static_camera_shots": 0, "motion_clips": 7}

        with patch("studio.v6_production.check_clip", side_effect=verified_clip):
            with patch("studio.hybrid_master.produce", side_effect=fake_native):
                result = render(self.project, self.review, self.takes, self.root, "review.mp4")
        self.assertEqual(result["status"], "RENDERED_REVIEW_ONLY")
        self.assertFalse(result["human_final_watch_passed"])
        self.assertFalse(result["published"])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
