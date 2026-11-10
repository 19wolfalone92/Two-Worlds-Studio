"""No-GPU tests for actual pixel decoding, false-ready media and image diagnostics."""
from __future__ import annotations
import io
import unittest

from PIL import Image,ImageDraw,ImageFilter

from studio.art_image_qc import InvalidArtwork, inspect_image_bytes, near_duplicate_checks


def make(kind="pattern",size=(256,448)):
    image=Image.new("RGB",size,(0,0,0))
    brush=ImageDraw.Draw(image)
    if kind=="pattern":
        for x in range(0,size[0],6):
            brush.rectangle((x,0,x+2,size[1]-1),fill=(255,255,255))
        for y in range(0,size[1],35):
            brush.ellipse((20,y,45,y+16),fill=(110,200,90))
    elif kind=="white":
        image.paste((255,255,255),(0,0,*size))
    elif kind=="black":
        pass
    output=io.BytesIO()
    image.save(output,format="PNG")
    return output.getvalue()

class ImageQualityTests(unittest.TestCase):
    def test_decoded_real_image_and_sha(self):
        result=inspect_image_bytes(make(),expected_size=(256,448))
        self.assertTrue(result["pixel_decode_verified"])
        self.assertTrue(result["technical_pass"])
        self.assertFalse(result["identity_verified"])
        self.assertTrue(result["art_review_required"])
        self.assertEqual(len(result["sha256"]),64)
        self.assertEqual(len(result["perceptual_dhash"]),16)

    def test_blank_black_white_frames_are_not_marked_technical_ready(self):
        black=inspect_image_bytes(make("black"))
        white=inspect_image_bytes(make("white"))
        self.assertFalse(black["technical_pass"])
        self.assertFalse(white["technical_pass"])
        self.assertIn("nearly_all_black",black["technical_issues"])
        self.assertIn("nearly_all_white",white["technical_issues"])

    def test_bad_signature_or_truncated_real_png_rejected(self):
        data=make()
        with self.assertRaises(InvalidArtwork):
            inspect_image_bytes(data[:220],expected_size=(256,448))
        with self.assertRaises(InvalidArtwork):
            inspect_image_bytes(b"fake PNG payload".ljust(1024,b"a"))

    def test_wrong_size_rejected(self):
        with self.assertRaises(InvalidArtwork):
            inspect_image_bytes(make(),expected_size=(720,1280))

    def test_required_focus_checks_only_opaque_subject_detail(self):
        image = Image.open(io.BytesIO(make())).convert("RGBA")
        box = [.2, .2, .8, .8]
        self.assertTrue(inspect_image_bytes(make(), focus_box=box)["required_focus"]["pass"])
        soft = image.filter(ImageFilter.GaussianBlur(8))
        image.putalpha(0)
        for bad in (soft, image):
            output = io.BytesIO()
            bad.save(output, format="PNG")
            report = inspect_image_bytes(output.getvalue(), focus_box=box)
            self.assertFalse(report["technical_pass"])
            self.assertIn("required_region_out_of_focus_or_transparent", report["technical_issues"])

    def test_advisory_duplicate_detection_without_shot_approval(self):
        a=inspect_image_bytes(make())
        b=inspect_image_bytes(make())
        results=near_duplicate_checks([a,b])
        self.assertEqual(len(results),1)
        self.assertTrue(results[0]["manual_duplicate_review_required"])
        self.assertEqual(results[0]["hamming_distance"],0)
        with self.assertRaises(ValueError):
            near_duplicate_checks([a],max_distance=-1)


if __name__=="__main__":
    unittest.main()
