import unittest
from studio.eb01_echo_production_manifest import plan
from studio.eb01_v7_storyboard import VOICE_SECONDS
class EchoProductionManifestTests(unittest.TestCase):
 def test_thirty_distinct_jobs_keep_15_15_and_all_original_voice_clocks(self):
  p=plan()
  self.assertEqual(p['job_allocation'],{'owner':15,'collaborator':15})
  self.assertEqual(len({j['job_id'] for j in p['jobs']}),30)
  self.assertEqual(p['unique_final_native_frames'],1027)
  self.assertGreaterEqual(p['native_duration_seconds'],40)
  for s in p['scenes']:
   if s['scene_id'] in VOICE_SECONDS:
    n=s['dialogue_end_scene_frame_exclusive']-s['dialogue_start_scene_frame']
    self.assertGreaterEqual(n/25,VOICE_SECONDS[s['scene_id']])
    self.assertLessEqual(s['dialogue_end_scene_frame_exclusive'],s['native_frames'])
 def test_scene_dependencies_never_cross_accounts_or_duplicate_overlap(self):
  p=plan();byid={j['job_id']:j for j in p['jobs']}
  for s in p['scenes']:
   total=0
   for k,jid in enumerate(s['job_ids']):
    j=byid[jid];self.assertEqual(j['native_frames']%4,1)
    if k:
     before=byid[j['predecessor_job_id']]
     self.assertEqual(before['account_alias'],j['account_alias'])
     self.assertEqual(before['end_scene_frame_exclusive']-1,j['start_scene_frame'])
    else:self.assertIsNone(j['predecessor_job_id'])
    total+=j['native_frames']-(1 if k else 0)
   self.assertEqual(total,s['native_frames'])
 def test_no_quality_flag_or_receipt_schema_can_submit_production_by_itself(self):
  p=plan();self.assertFalse(p['production_execution_authorized'])
  self.assertEqual(p['gpu_submissions'],0)
  self.assertIn('AUDIO_CONDITIONED_CONTINUATION_PILOT_REQUIRED',p['production_blockers'])
  self.assertFalse(p['two_simultaneous_generators_per_account_approved'])
