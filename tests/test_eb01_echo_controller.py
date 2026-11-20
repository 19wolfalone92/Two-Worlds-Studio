"""Meaningful duplicate-work and private-source isolation regressions."""
import base64,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools.eb01_echo_kaggle_pilot import prepare,run_pilot
from tools.eb01_native_two_account_pilot import PilotBlocked

class EchoControllerTests(unittest.TestCase):
    def test_two_immutable_queues_have_different_characters_and_preserve_whole_voices(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);kernels=[]
            for alias in ('owner','collaborator'):
                dest=root/alias;dest.mkdir()
                folder,kernel=prepare(dest,alias,'exampleuser','1234567','a'*40,{})
                kernels.append(kernel)
                meta=json.loads((folder/'kernel-metadata.json').read_text())
                self.assertEqual(meta['is_private'],'true')
                self.assertEqual(meta['dataset_sources'],['wolf19loner92/eb01-v7-private-production-sources'])
                self.assertNotIn('TOKEN',(folder/'kernel_main.py').read_text().split('CONFIG_B64=')[1].splitlines()[0])
            self.assertNotEqual(*kernels)
    def test_retry_and_explicit_collection_cannot_resubmit_gpu_work(self):
        for attempt,collect in ((2,None),(1,'1234567')):
            with self.subTest(attempt=attempt),tempfile.TemporaryDirectory() as temp:
                with patch('tools.eb01_echo_kaggle_pilot.query',return_value=('AUTHENTICATED_READ_ONLY','exampleuser')):
                    with patch('tools.eb01_echo_kaggle_pilot.call',side_effect=PilotBlocked('STOP_TEST')) as api:
                        report=run_pilot('owner','private',Path(temp),'1234567','a'*40,attempt,{},collect_run=collect)
                self.assertEqual(report['gpu_submission_attempts'],0)
                self.assertEqual(api.call_args.args[1][2],'status')
    def test_wrong_voice_receipt_cannot_become_quality_approval(self):
        # A no-MP4 setup failure is categorized without leaking raw diagnostic text.
        from tools.eb01_echo_kaggle_pilot import failure_category
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'EB01_ECHO_PRIVATE_NATIVE.log').write_text('private portrait /secret; CUDA out of memory')
            self.assertEqual(failure_category(p),'CUDA_OUT_OF_MEMORY')

if __name__=='__main__':unittest.main()
