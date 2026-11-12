"""Offline tests - no Kaggle access, GPU, image generation, or public source art."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from studio.eb01_canon_kaggle import ARCHIVE, CONTACT, RECEIPT, MODEL, prepare, verify


class EB01CanonPrivateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.nonce="eb01-canon-local-test-1"

    def fake_output(self):
        (self.root/"canon").mkdir()
        img=Image.new("RGB",(704,1216),"#a27751")
        images={}
        archive=self.root/ARCHIVE
        with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
            for who in ("KOSTYA","ELIZABETH"):
                path=self.root/"canon"/(who+".png")
                img.save(path)
                data=path.read_bytes()
                z.write(path,"canon/"+who+".png")
                images[who]={
                    "sha256":hashlib.sha256(data).hexdigest(),"size":len(data),
                    "width":704,"height":1216,"art_reviewed":False}
        preview=self.root/CONTACT
        Image.new("RGB",(484,483),"#7f654b").save(preview,quality=80)
        receipt={
            "success":True,"nonce":self.nonce,"model":MODEL,
            "private":True,"gpu_inference_performed":True,
            "video_generated":False,"published":False,"image_identity_verified":False,
            "full_video_allowed":False,"generated_images":images,
            "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
            "contact_sha256":hashlib.sha256(preview.read_bytes()).hexdigest()}
        (self.root/RECEIPT).write_text(json.dumps(receipt),encoding="utf-8")
        return receipt

    def test_private_worker_preparation_requires_no_gpu(self):
        with tempfile.TemporaryDirectory() as source:
            root=Path(source)
            (root/"tools").mkdir()
            (root/"tools"/"kaggle_eb01_canon_entry.py").write_text(
                "# __EB01_CANON_CONFIG_B64__\n")
            result=prepare("wolf19loner92",self.nonce,root/"kernel",root=root)
            self.assertEqual(result["max_images"],2)
            self.assertFalse(result["gpu_executed"])
            self.assertTrue(result["output_private"])
            meta=json.loads((root/"kernel"/"kernel-metadata.json").read_text())
            self.assertEqual(meta["is_private"],"true")
            self.assertEqual(meta["enable_gpu"],"true")
            self.assertNotIn("KAGGLE_API_TOKEN",(root/"kernel"/"kernel_main.py").read_text())

    def test_verified_private_review_does_not_approve_art(self):
        self.fake_output()
        result=verify(self.root,self.nonce)
        self.assertEqual(result["status"],"TWO_PRIVATE_REAL_CANDIDATES_REQUIRE_HUMAN_REVIEW")
        self.assertFalse(result["visual_identity_approved"])
        self.assertFalse(result["video_generated"])

    def test_bad_canon_sha_is_blocked(self):
        data=self.fake_output()
        data["generated_images"]["KOSTYA"]["sha256"]="0"*64
        (self.root/RECEIPT).write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"source-locked"):
            verify(self.root,self.nonce)

    def test_bad_run_nonce_is_blocked(self):
        self.fake_output()
        with self.assertRaisesRegex(ValueError,"Not valid"):
            verify(self.root,"wrong-nonce")

    def test_missing_original_package_is_blocked(self):
        self.fake_output()
        (self.root/ARCHIVE).unlink()
        with self.assertRaises(FileNotFoundError):
            verify(self.root,self.nonce)


if __name__=="__main__":
    unittest.main()
