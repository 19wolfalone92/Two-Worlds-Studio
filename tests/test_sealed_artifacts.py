"""Private drawing transport: authenticated bytes, no plaintext in public output."""
import tempfile
import unittest
from pathlib import Path
from studio.sealed_artifacts import generate_keys,seal,unseal

try:
    import cryptography
except ImportError:
    cryptography=None


@unittest.skipUnless(cryptography,"Optional private-output cryptography dependency not installed")
class SealedArtTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.private,self.public=self.root/'private.pem',self.root/'public.pem'
        generate_keys(self.private,self.public)
        self.source=self.root/'original.zip';self.source.write_bytes(b'PRIVATE_DRAWING_BYTES_FOR_SYNTHETIC_TEST'*100)
        self.sealed=self.root/'result.twenc';self.output=self.root/'returned.zip'
        self.context={'run_nonce':'unit-run-1','source_sha':'a'*40}

    def test_round_trip_is_private_and_bound_to_the_request(self):
        info=seal(self.source,self.public,self.sealed,self.context)
        self.assertNotIn(b'PRIVATE_DRAWING_BYTES',self.sealed.read_bytes())
        self.assertNotIn(b'PRIVATE KEY',self.public.read_bytes())
        self.assertEqual(self.private.stat().st_mode&0o777,0o600)
        self.assertFalse(info['raw_art_uploaded'])
        self.assertTrue(unseal(self.sealed,self.private,self.output,self.context)['authenticated'])
        self.assertEqual(self.output.read_bytes(),self.source.read_bytes())

    def test_wrong_generation_context_never_writes_plaintext(self):
        seal(self.source,self.public,self.sealed,self.context)
        with self.assertRaisesRegex(ValueError,'another generation'):
            unseal(self.sealed,self.private,self.output,{'run_nonce':'wrong','source_sha':'a'*40})
        self.assertFalse(self.output.exists())

    def test_tampered_envelope_is_rejected_before_plaintext_write(self):
        from cryptography.exceptions import InvalidTag
        seal(self.source,self.public,self.sealed,self.context)
        data=bytearray(self.sealed.read_bytes());data[-1]^=1;self.sealed.write_bytes(data)
        with self.assertRaises(InvalidTag):unseal(self.sealed,self.private,self.output,self.context)
        self.assertFalse(self.output.exists())

    def test_existing_recipient_or_returned_art_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):generate_keys(self.private,self.public)
        seal(self.source,self.public,self.sealed,self.context)
        with self.assertRaises(FileExistsError):seal(self.source,self.public,self.sealed,self.context)
        self.output.write_bytes(b'existing owner file')
        with self.assertRaises(FileExistsError):unseal(self.sealed,self.private,self.output,self.context)
        self.assertEqual(self.output.read_bytes(),b'existing owner file')

    def test_public_workflow_uploads_only_the_sealed_container(self):
        root=Path(__file__).resolve().parent.parent
        workflow=(root/'.github/workflows/character-factory.yml').read_text()
        upload=workflow.split('uses: actions/upload-artifact@v4')[1]
        self.assertIn('path: dist/private_return/TW_FACTORY_RESULT.twenc',upload)
        self.assertNotIn('TWO_WORLDS_ARTIFACTS.zip',upload)
        self.assertIn("head.repo.full_name == github.repository",workflow)
        self.assertIn("head.ref == 'feat/from-scratch-character-factory'",workflow)

    def test_worker_installs_drawing_imports_before_request_validation(self):
        root=Path(__file__).resolve().parent.parent
        workflow=(root/'.github/workflows/character-factory.yml').read_text()
        self.assertLess(workflow.index('name: Install official worker tools'),
                        workflow.index('name: Validate explicit bounded generation'))
        self.assertIn("always() && env.KERNEL_HANDLE != ''",workflow)


if __name__=='__main__':unittest.main()
