import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from studio import eb01_native_t4_profile as m
from tools.eb01_native_two_account_pilot import prepare


class NativeProfileTests(unittest.TestCase):
    def test_reference_comes_from_full_original_and_preserves_native_pixels(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            source=root/'KOSTYA.png'
            Image.new('RGB',(704,1216),(50,90,120)).save(source)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.dict(m.SRC_SHA, {'KOSTYA':digest}):
                report=m.native_reference(root,root/'native.png','S02')
            with Image.open(root/'native.png') as image:
                self.assertEqual(image.size,(576,1024))
            self.assertFalse(report['low_resolution_reference_upscaled'])
            self.assertGreaterEqual(report['source_crop_size'][0],576)
            self.assertGreaterEqual(report['source_crop_size'][1],1024)

    def test_modified_original_blocks_before_reference_creation(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            Image.new('RGB',(704,1216)).save(root/'KOSTYA.png')
            with self.assertRaises(ValueError):
                m.native_reference(root,root/'out.png','S02')
            self.assertFalse((root/'out.png').exists())

    def test_pilot_packaging_is_private_pinned_and_distinct_by_account(self):
        ids=[]
        for alias,handle in [('owner','first'),('collaborator','second')]:
            with tempfile.TemporaryDirectory() as t:
                folder,kernel=prepare(Path(t),alias,handle,'43920000000','a'*40)
                ids.append(kernel)
                meta=json.loads((folder/'kernel-metadata.json').read_text())
                self.assertEqual(meta['is_private'],'true')
                self.assertEqual(meta['machine_shape'],'NvidiaTeslaT4')
                self.assertEqual(len(meta['dataset_sources']),1)
                code=(folder/'kernel_main.py').read_text()
                ast.parse(code)
                self.assertNotIn('__EB01_LTX2B_CONFIG_B64__',code)
                self.assertNotIn('KAGGLE_API_TOKEN',code)
        self.assertNotEqual(ids[0].split('/')[1],ids[1].split('/')[1])

    def test_exact_patch_refuses_unknown_upstream(self):
        with self.assertRaises(ValueError):
            m.replace_once('unknown source','required marker','update')
        with self.assertRaises(ValueError):
            m.replace_once('twice twice','twice','update')


if __name__=='__main__': unittest.main()
