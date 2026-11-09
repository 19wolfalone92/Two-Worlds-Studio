from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from studio.sealed_artifacts import generate_keys,unseal
from tools.batch03_production_relay import seal_return


class ReturnEnvelopeTests(unittest.TestCase):
    def test_large_batch_roundtrips_authenticated_parts_in_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);private=root/'private.pem';public=root/'public.pem'
            generate_keys(private,public)
            archive=root/'media.zip';archive.write_bytes(bytes(range(256))*5)
            with patch('tools.batch03_production_relay.TEMP',root):
                seal_return(archive,public,root/'return','test-nonce',part_bytes=400)
            manifest=json.loads((root/'return/return_parts.json').read_text())
            joined=b''
            for part in manifest['parts']:
                dest=root/f"decoded_{part['index']}"
                unseal(root/'return'/part['file'],private,dest,
                       {'purpose':'batch03-production','nonce':'test-nonce',
                        'part':part['index'],'count':manifest['count'],
                        'archive_sha256':manifest['archive_sha256']})
                joined+=dest.read_bytes()
            self.assertEqual(joined,archive.read_bytes())
            self.assertEqual(hashlib.sha256(joined).hexdigest(),manifest['archive_sha256'])
