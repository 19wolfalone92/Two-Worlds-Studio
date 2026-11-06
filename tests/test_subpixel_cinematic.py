"""Regression: layered 2.5D motion uses fractional crop coordinates, not pixel jumps."""
from __future__ import annotations
import unittest
from PIL import Image,ImageDraw
from studio.cinematic import _place,camera_state


class SubpixelParallaxTests(unittest.TestCase):
    def test_fractional_translation_changes_frames_smoothly(self):
        background=Image.new("RGBA",(120,160),(0,0,0,255))
        striped=Image.new("RGBA",(120,160),(0,0,0,255))
        brush=ImageDraw.Draw(striped)
        for x in range(0,120,6):
            brush.rectangle((x,0,x+2,159),fill=(255,255,255,255))
        left=_place(background,striped,zoom=1.28,cx=0.01,cy=0,
                    depth=0.5,role="character")
        right=_place(background,striped,zoom=1.28,cx=0.02,cy=0,
                     depth=0.5,role="character")
        self.assertEqual(left.size,background.size)
        self.assertEqual(right.size,background.size)
        # Old integer x/y crop rounded the two subpixel positions to same frame.
        self.assertNotEqual(left.tobytes(),right.tobytes())

    def test_subpixel_transform_preserves_alpha_and_eased_camera(self):
        background=Image.new("RGBA",(120,160),(0,0,0,255))
        sprite=Image.new("RGBA",(120,160),(0,0,0,0))
        ImageDraw.Draw(sprite).rectangle((30,30,90,130),fill=(200,70,40,255))
        result=_place(background,sprite,zoom=1.25,cx=0.125,cy=-0.12,
                      depth=0.6,role="character")
        self.assertEqual(result.mode,"RGBA")
        self.assertEqual(result.size,(120,160))
        self.assertLess(result.getchannel("A").getextrema()[0],255)
        scene={"cinematic":{"camera":{"start":{"x":-.25,"y":0,"zoom":1.1},
                                     "end":{"x":.25,"y":.05,"zoom":1.2}}}}
        self.assertEqual(camera_state(scene,0.5)[0],1.15)
        self.assertLess(camera_state(scene,0.25)[0],camera_state(scene,0.75)[0])

if __name__=="__main__":
    unittest.main()
