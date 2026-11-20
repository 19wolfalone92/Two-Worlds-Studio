import json,unittest
from tools.eb01_kaggle_quota_observer import parse_quota
class QuotaObserverTests(unittest.TestCase):
 def test_actual_hours_are_not_misrepresented_as_pooled_memory(self):
  r=parse_quota(json.dumps([{'resource':'GPU','used':'4.25h','remaining':'25.75h','total':'30.00h','refreshAt':'2026-11-23T00:00:00+00:00'}]))
  self.assertEqual(r['GPU']['remaining_hours'],25.75)
 def test_invalid_or_missing_actual_quota_fails_closed(self):
  for raw in ('[]','{}','[{"resource":"GPU","used":"NaNh","remaining":"30h","total":"30h"}]'):
   with self.subTest(raw=raw),self.assertRaises((ValueError,KeyError)):parse_quota(raw)
