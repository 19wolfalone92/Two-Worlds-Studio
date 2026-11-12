"""Independent, source-backed SHA lock regression: catches 63-character manual typo.

The authoritative seven reference-image hashes were read from a completed
private CPU Kaggle storyboard receipt. Only hashes are stored publicly.
"""
import json
import re
import unittest
from pathlib import Path

from studio.eb01_v7_storyboard import SRC_SHA,SHOTS,VOICE_SHA
from tools.kaggle_eb01_v7_full_entry import REVIEWED_SOURCE_SHA

class E01SevenGenuineSourceHashes(unittest.TestCase):
 def test_exact_cpu_storyboard_receipt_matches_pinned_worker_sha(self):
  source=Path(__file__).parent/"fixtures"/"eb01_v7_original_private_contact_sha.json"
  expected=json.loads(source.read_text(encoding="utf-8"))
  self.assertEqual(expected["schema"],7)
  self.assertEqual(expected["frames"],REVIEWED_SOURCE_SHA)
  self.assertEqual(expected["private_source_sha256"],SRC_SHA)
  self.assertEqual(set(REVIEWED_SOURCE_SHA),{shot[0] for shot in SHOTS})
  for sid,value in REVIEWED_SOURCE_SHA.items():
   with self.subTest(shot=sid):
    self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}",value))
  self.assertEqual(len(VOICE_SHA),6)

if __name__=="__main__":
 unittest.main()
