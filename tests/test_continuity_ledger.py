"""CPU-only visual continuity ledger checks; never pretend to recognize faces."""
import unittest

from studio.continuity_ledger import audit

A = "a" * 64
B = "b" * 64
C = "c" * 64

def scene(name, *, wardrobe="coat", variant="adult",
          location_hash=A, person_hash=B, prop_hash=C,
          transitions=None):
    return {
        "scene_id": name,
        "continuity": {
            "location": {"id": "kitchen", "layout_sha256": location_hash},
            "cast": [{
                "body_id": "Nina", "age_variant": variant,
                "reference_sha256": person_hash, "wardrobe_id": wardrobe,
            }],
            "props": [{"id": "blue_box", "reference_sha256": prop_hash}],
            "transitions": transitions or [],
        }
    }


class ContinuityLedger(unittest.TestCase):
    def test_consistent_reference_anchors_across_three_shots(self):
        result = audit([scene("P1S1"), scene("P1S2"), scene("P2S1")], strict=True)
        self.assertTrue(result["structural_continuity_pass"])
        self.assertEqual(result["metadata_coverage"], 3)
        self.assertEqual(result["character_anchors"], 1)
        self.assertFalse(result["actual_face_similarity_verified"])
        self.assertFalse(result["actual_location_appearance_verified"])

    def test_rejects_actor_identity_drift(self):
        with self.assertRaisesRegex(ValueError, "visual identity"):
            audit([scene("P1S1"), scene("P1S2", person_hash=A)])

    def test_rejects_set_geometry_and_prop_drift(self):
        with self.assertRaisesRegex(ValueError, "geometry drift"):
            audit([scene("P1S1"), scene("P1S2", location_hash=B)])
        with self.assertRaisesRegex(ValueError, "prop blue_box"):
            audit([scene("P1S1"), scene("P1S2", prop_hash=B)])

    def test_wardrobe_and_age_changes_require_explicit_reviewed_reason(self):
        with self.assertRaisesRegex(ValueError, "costume changed"):
            audit([scene("P1S1"), scene("P1S2", wardrobe="dress")])
        transition = [{"kind": "wardrobe", "id": "Nina",
                       "reason": "The character changes clothes after arriving home.",
                       "reviewed_by": "editor"}]
        self.assertTrue(audit([scene("P1S1"),
                               scene("P1S2", wardrobe="dress",
                                     transitions=transition)], strict=True)["structural_continuity_pass"])
        with self.assertRaisesRegex(ValueError, "age variant"):
            audit([scene("P1S1"), scene("P2S1", variant="child")])
        transition = [{"kind": "age_variant", "id": "Nina",
                       "reason": "The second scene is an approved flashback.",
                       "reviewed_by": "editor"}]
        self.assertEqual(audit([scene("P1S1"), scene("P2S1", variant="child",
                      transitions=transition)], strict=True)["character_anchors"], 2)

    def test_unreviewed_metadata_missing_is_reported_not_fabricated(self):
        data = [scene("P1S1"), {"scene_id": "P1S2"}]
        result = audit(data)
        self.assertEqual(result["unverifiable_shots"], 1)
        self.assertFalse(result["structural_continuity_pass"])
        with self.assertRaisesRegex(ValueError, "no continuity metadata"):
            audit(data, strict=True)


if __name__ == "__main__":
    unittest.main()
