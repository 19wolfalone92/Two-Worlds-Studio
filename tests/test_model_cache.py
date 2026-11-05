from pathlib import Path
import hashlib,tempfile,unittest
from studio.model_cache import stage_cached_tree
import shutil


class ModelStagingTests(unittest.TestCase):
    def test_stream_copy_preserves_bytes_and_ignores_partial_downloads(self):
        with tempfile.TemporaryDirectory() as folder:
            src=Path(folder);(src/'model.safetensors').write_bytes(bytes(range(256))*17)
            (src/'.cache').mkdir();(src/'.cache/unfinished.incomplete').write_bytes(b'partial')
            target,receipt=stage_cached_tree(src,max_bytes=10000)
            try:
                self.assertEqual((target/'model.safetensors').read_bytes(),(src/'model.safetensors').read_bytes())
                self.assertFalse((target/'.cache').exists())
                self.assertEqual(receipt['files'][0]['sha256'],hashlib.sha256((src/'model.safetensors').read_bytes()).hexdigest())
            finally:shutil.rmtree(target)

    def test_staging_cannot_exceed_the_declared_disk_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            src=Path(folder);(src/'weights').write_bytes(b'123456789')
            with self.assertRaises(ValueError):stage_cached_tree(src,max_bytes=8)
