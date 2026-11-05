"""No-model tests: character animation quality is not a camera pan."""
import unittest

from studio.animation_quality import audit_performance


class CharacterMotionQuality(unittest.TestCase):
    def test_camera_only_sticker_cannot_be_master_approved(self):
        ep = {
            "id": "P01", "scenes": [
                {"id": "S01", "cinematic": {"layers": [{"role": "character", "body_id": "Nina"}]},
                 "animation": {"kind": "intentional_hold", "hold_reason": "short",
                               "blink_style": "overlay_only"}}
            ],
        }
        report = audit_performance(ep, publish=False)
        self.assertFalse(report["master_ready"])
        self.assertGreater(report["problems"], 0)
        self.assertFalse(report["automatic_aesthetic_certification"])
        with self.assertRaisesRegex(ValueError, "cinematic performance"):
            audit_performance(ep, publish=True)

    def test_authored_rig_needs_environment_and_reviewer(self):
        scene = {
            "id": "S01",
            "cinematic": {"layers": [{"role": "character",
                                      "rig": {"tracks": {"head_turn": [0, 5, 12]}}}]},
            "animation": {
                "kind": "rigged_2d", "cues": ["expression", "head_turn"],
                "integration": {"contact_shadow": True, "lighting": True,
                                "perspective": True},
                "reviewed_by": "owner"
            }
        }
        result = audit_performance({"scenes": [scene]}, publish=True)
        self.assertTrue(result["master_ready"])
        self.assertFalse(result["automatic_aesthetic_certification"])


if __name__ == "__main__":
    unittest.main()
