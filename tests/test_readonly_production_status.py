import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from studio.sealed_artifacts import generate_keys,unseal
from tools import batch03_production_relay as relay
import zipfile

class ReadOnlyStatusTest(unittest.TestCase):
    def test_status_never_submits_or_downloads_a_gpu_job(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);runtime=root/'temp';runtime.mkdir()
            key=root/'private.pem';public=root/'.github/batch03_requests/recipient_public.pem'
            generate_keys(key,public)
            config={'nonce':'test-status','phase':'status','story':'13','kernels':['wolf19loner92/tw-batch03-existing-job']}
            with patch.object(relay,'ROOT',root),patch.object(relay,'TEMP',runtime),patch.object(relay,'request',return_value=config),patch.object(relay,'command',side_effect=['free quota','RUNNING']) as command:
                relay.produce()
            self.assertEqual([x.args[0] for x in command.call_args_list],[['kaggle','quota'],['kaggle','kernels','status',config['kernels'][0]]])
            out=root/'status.zip';unseal(root/'dist/private_batch03/production_result.twenc',key,out,{'purpose':'batch03-production','nonce':'test-status'})
            with zipfile.ZipFile(out) as z:summary=json.loads(z.read('relay_summary.json'))
            self.assertIs(summary['gpu_requested'],False)
            self.assertIs(summary['paid_services'],False)
