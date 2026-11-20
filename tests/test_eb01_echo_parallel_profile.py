import unittest
from studio.eb01_echo_parallel_profile import specs
class SingleCardParallelProfileTests(unittest.TestCase):
 def test_full_original_voices_fit_both_separate_card_workers(self):
  seeds=set()
  for account in ('owner','collaborator'):
   for slot,s in enumerate(specs(account)):
    self.assertEqual(s['physical_gpu_index'],slot)
    self.assertGreaterEqual(s['frames']/25+1e-6,s['voice_seconds']+.2)
    self.assertNotIn(s['seed'],seeds);seeds.add(s['seed'])
    self.assertFalse(s['whole_movie_approved'])
 def test_continuation_overlaps_are_counted_once(self):
  for account in ('owner','collaborator'):
   for s in specs(account):
    self.assertEqual(sum(n-1 for n in s['segment_frames'])+1,s['frames'])
    for n in s['segment_frames']:self.assertEqual(n%4,1)
