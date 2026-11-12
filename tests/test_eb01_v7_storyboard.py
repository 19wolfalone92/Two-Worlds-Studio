"""Source-locked EB01 storyboard generation requires exactly 2 original portraits/6 WAV."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from studio.eb01_v7_storyboard import (SHOTS,SRC_SHA,VOICE_SHA,
                                       DIALOGUES,WIDTH,HEIGHT,FPS,
                                       gather_private,contact_sheet)

class E01V7SourceLockTests(unittest.TestCase):
 def test_seven_real_character_refs_and_voice_clocks(self):
  self.assertEqual(len(SHOTS),7)
  self.assertEqual(set(DIALOGUES),set(VOICE_SHA))
  self.assertEqual((WIDTH,HEIGHT,FPS),(320,576,24))
  self.assertEqual([s[0] for s in SHOTS],["S00","S01","S02","S03","S04","S05","S06"])
  for sid,who,frames,seconds,zoom,cx,cy,prompt in SHOTS:
   self.assertIn(who,SRC_SHA)
   self.assertEqual((frames-1)%4,0)
   self.assertGreaterEqual(frames/FPS,seconds)
   self.assertLessEqual(frames,129)
   self.assertGreaterEqual(len(prompt),120)
 def test_unique_source_assets_never_created_from_random_actors(self):
  self.assertEqual(len(SRC_SHA),2)
  self.assertTrue(all(len(s)==64 for s in SRC_SHA.values()))
  self.assertEqual(len(VOICE_SHA),6)
  self.assertNotEqual(SRC_SHA["KOSTYA"],SRC_SHA["ELIZABETH"])
 def test_missing_private_dataset_blocks_generation(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   with self.assertRaisesRegex(ValueError,"Canonical KOSTYA"):
    gather_private(p,p/"output")
 def test_contact_does_not_create_original_art(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   (root/"frames").mkdir()
   for sid,*rest in SHOTS:
    Image.effect_noise((WIDTH,HEIGHT),31).convert("RGB").save(root/"frames"/(sid+".png"))
   dest=root/"contact.jpg"
   contact_sheet(root,dest)
   with Image.open(dest) as img:
    self.assertEqual(img.size,(720,736))
   with self.assertRaisesRegex(FileExistsError,"overwrite"):
    contact_sheet(root,dest)

if __name__=="__main__":unittest.main()
