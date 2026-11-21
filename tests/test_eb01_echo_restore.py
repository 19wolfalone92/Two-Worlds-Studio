"""Recovery cannot silently run diffusion or accept altered latent bytes."""
import base64,hashlib,json,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools.eb01_echo_restore_checkpoint import checkpoint_info
from tools.eb01_echo_restore_pilot import prepare,run
from tools.eb01_native_two_account_pilot import PilotBlocked

class RestoreTests(unittest.TestCase):
    def test_frozen_tensor_header_and_payload_are_sha_bound_without_pickle(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'latents.safetensors'
            doc={'latents':{'dtype':'F16','shape':[1,16,17,128,72],'data_offsets':[0,5013504]}}
            header=json.dumps(doc).encode();payload=struct.pack('<Q',len(header))+header+bytes(5013504)
            p.write_bytes(payload);digest=hashlib.sha256(payload).hexdigest()
            self.assertEqual(checkpoint_info(p,digest)['new_denoising_steps'],0)
            p.write_bytes(payload[:-1]+b'x')
            with self.assertRaises(ValueError):checkpoint_info(p,digest)

    def test_private_recovery_attaches_original_and_never_invokes_denoiser(self):
        with tempfile.TemporaryDirectory() as t:
            folder,kernel,original=prepare(Path(t),'exampleuser','123456789','a'*40)
            meta=json.loads((folder/'kernel-metadata.json').read_text());source=(folder/'kernel_main.py').read_text()
            self.assertEqual(meta['is_private'],'true');self.assertEqual(meta['kernel_sources'],[original])
            self.assertNotIn('tools.eb01_echo_native_infer',source)
            self.assertIn('tools.eb01_echo_restore_checkpoint',source);self.assertIn('deadline_seconds=900',source)
            self.assertNotEqual(kernel,original)

    def test_retry_only_collects_same_job_and_never_pushes_again(self):
        with tempfile.TemporaryDirectory() as t,patch('tools.eb01_echo_restore_pilot.query',return_value=('AUTHENTICATED_READ_ONLY','exampleuser')):
            with patch('tools.eb01_echo_restore_pilot.call',side_effect=PilotBlocked('STOP_TEST')) as api:
                report=run('private',Path(t),'123456789','a'*40,2)
        self.assertEqual(report['gpu_submission_attempts'],0)
        self.assertEqual(api.call_args.args[1][2],'status')
