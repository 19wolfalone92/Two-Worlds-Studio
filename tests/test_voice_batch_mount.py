from pathlib import Path
import tempfile
import unittest
from tools.kaggle_voice_batch_entry import resolve_source


class VoiceBatchMountTests(unittest.TestCase):
    def test_real_kaggle_unpacked_dataset_layout_is_supported(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data=root/'datasets'/'owner'/'two-worlds-private-batch03-20261104'
            for i in range(13,19):
                p=data/'production_queue'/str(i)/'project.json';p.parent.mkdir(parents=True);p.write_text('{}')
            self.assertEqual(resolve_source(root/'missing.zip',data.name,root),data)

    def test_other_kernel_output_is_not_confused_with_original_dataset(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'other-kernel'/'production_queue'/'13'/'project.json'
            p.parent.mkdir(parents=True);p.write_text('{}')
            with self.assertRaises(FileNotFoundError):
                resolve_source(root/'missing.zip','two-worlds-private-batch03-20261104',root)
