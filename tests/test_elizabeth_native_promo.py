"""Native Elizabeth Bones review contract; no cloud/editor dependencies."""
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from tools.elizabeth_native_promo import SHOTS, manifest_for


class NativeElizabethPreviewTests(unittest.TestCase):
    def test_locked_six_turns_and_timing(self):
        self.assertEqual([r[0] for r in SHOTS], ["S00","S01","S02","S03","S04","S05","S06"])
        self.assertEqual(len([r for r in SHOTS if r[2]]), 6)
        self.assertAlmostEqual(sum(r[1] for r in SHOTS), 18.475, places=3)
        self.assertIn("Двести три", SHOTS[4][2])
        self.assertIn("На испытательном", SHOTS[-1][2])

    def test_unvoiced_preview_is_explicit_and_full_voice_gate_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for sid, *_ in SHOTS:
                Image.new("RGB",(576,1024),(40,32,37)).save(root/(sid+".png"))
            plan,missing=manifest_for(root,False)
            self.assertEqual(missing, ["S01","S02","S03","S04","S05","S06"])
            self.assertFalse(plan["editorial_approved"])
            self.assertFalse(plan["rights_approved"])
            self.assertEqual(plan["language"], "ru")
            self.assertTrue(plan["burn_subtitles"])
            with self.assertRaisesRegex(ValueError, "Not a full-voice preview"):
                manifest_for(root,True)

    def test_no_missing_image_substitution(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "never synthesize"):
                manifest_for(Path(temp),False)


if __name__ == "__main__":
    unittest.main()
