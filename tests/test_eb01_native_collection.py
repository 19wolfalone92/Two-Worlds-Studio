import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import eb01_native_two_account_pilot as m


class ExistingNativeCollectionTests(unittest.TestCase):
    def test_collect_mode_never_resubmits_even_after_original_failure(self):
        calls=[]
        def api(token,args,**kwargs):
            calls.append(args)
            return 'KernelWorkerStatus.ERROR' if args[2]=='status' else ''
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);job=root/'private';job.mkdir()
            with patch.object(m,'query',return_value=('AUTHENTICATED_READ_ONLY','first')), \
                 patch.object(m,'call',side_effect=api), \
                 patch.dict(m.os.environ,{'RUNNER_TEMP':t}):
                report=m.run_pilot('owner','selected',job,'44699999999','a'*40,2,
                                   collect_run='43918550500',poll_seconds=1)
            self.assertEqual(report['gpu_submission_attempts'],0)
            self.assertFalse(report['native_output']['actual_mp4_created'])
            self.assertTrue(report['encrypted_private_return_created'])
        self.assertTrue(calls)
        self.assertTrue(all(args[2] in ('status','output') for args in calls))

    def test_unknown_kernel_state_fails_closed_without_generation(self):
        with tempfile.TemporaryDirectory() as t:
            with patch.object(m,'query',return_value=('AUTHENTICATED_READ_ONLY','first')), \
                 patch.object(m,'call',return_value='unknown state') as api:
                report=m.run_pilot('owner','selected',Path(t),'44699999999','a'*40,2,
                                   collect_run='43918550500',poll_seconds=1)
        self.assertEqual(report['gpu_submission_attempts'],0)
        self.assertEqual(report['status'],'BLOCKED_UNKNOWN_KERNEL_STATE')
        self.assertEqual(api.call_count,1)


if __name__=='__main__':unittest.main()
