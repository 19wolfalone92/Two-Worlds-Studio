import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools.eb01_echo_parallel_pilot import prepare,run
from tools.eb01_native_two_account_pilot import PilotBlocked
QUOTA=json.dumps([{'resource':'GPU','used':'1h','remaining':'29h','total':'30h','refreshAt':'2026-11-21T00:00:00Z'}])
class ParallelControllerTests(unittest.TestCase):
 def test_two_accounts_get_different_private_benchmark_kernels(self):
  with tempfile.TemporaryDirectory() as t:
   roots=[Path(t)/'owner',Path(t)/'collaborator'];ids=[]
   for alias,root in zip(('owner','collaborator'),roots):
    root.mkdir();folder,kernel=prepare(root,alias,'exampleuser','1234567','a'*40,{})
    ids.append(kernel)
    self.assertEqual(json.loads((folder/'kernel-metadata.json').read_text())['is_private'],'true')
   self.assertNotEqual(*ids)
 def test_retry_only_collects_the_same_notebook_and_never_resubmits(self):
  with tempfile.TemporaryDirectory() as t,patch('tools.eb01_echo_parallel_pilot.query',return_value=('AUTHENTICATED_READ_ONLY','exampleuser')):
   with patch('tools.eb01_echo_parallel_pilot.call',side_effect=[PilotBlocked('STOP_TEST')]) as api:
    result=run('owner','private',Path(t),'1234567','a'*40,2,{})
   self.assertEqual(result['gpu_submission_attempts'],0)
   self.assertEqual(api.call_args.args[1][1:3], ['kernels','status'])
   self.assertEqual(api.call_count,1)
 def test_uncertain_push_is_not_automatically_attempted_twice(self):
  with tempfile.TemporaryDirectory() as t,patch('tools.eb01_echo_parallel_pilot.query',return_value=('AUTHENTICATED_READ_ONLY','exampleuser')):
   with patch('tools.eb01_echo_parallel_pilot.call',side_effect=[QUOTA,PilotBlocked('CLI_UNAVAILABLE_OR_TIMEOUT')]) as api:
    result=run('owner','private',Path(t),'1234567','a'*40,1,{})
   self.assertEqual(result['gpu_submission_attempts'],1)
   self.assertEqual(sum('push' in call.args[1] for call in api.call_args_list),1)
