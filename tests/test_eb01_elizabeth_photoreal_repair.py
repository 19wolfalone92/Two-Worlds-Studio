"""Offline targeted visual repair contracts: only Elizabeth gets re-generated."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from studio.eb01_canon_kaggle import ARCHIVE, CONTACT, RECEIPT, MIXED_MODEL, MODEL, verify as verify_canon
from studio.eb01_elizabeth_photoreal_repair import MODEL as PHOTO_MODEL, original_kostya, prepare, verify


class RepairOnlyElizabethTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.nonce="eb01canon-test-nonce"
        self.raws={}
        for who in ("KOSTYA","ELIZABETH"):
            path=self.root/(who+".png")
            Image.effect_noise((704,1216),36).convert("RGB").save(path)
            self.raws[who]=path.read_bytes()
        self.save(previous=True)

    def save(self,previous=False):
        with zipfile.ZipFile(self.root/ARCHIVE,"w",zipfile.ZIP_DEFLATED) as z:
            for who in self.raws:
                z.writestr("canon/"+who+".png",self.raws[who])
        Image.effect_noise((484,483),35).convert("RGB").save(self.root/CONTACT,quality=84)
        records={
            who:{
                "sha256":hashlib.sha256(self.raws[who]).hexdigest(),
                "size":len(self.raws[who]),"width":704,"height":1216,
                "art_reviewed":False,"model":MODEL if who=="KOSTYA" else PHOTO_MODEL,
                "reused_unchanged":who=="KOSTYA"
            } for who in self.raws
        }
        data={
            "nonce":self.nonce,"success":True,"private":True,
            "gpu_inference_performed":True,"video_generated":False,
            "published":False,"image_identity_verified":False,
            "full_video_allowed":False,
            "model":MODEL if previous else MIXED_MODEL,
            "parent_KOSTYA_sha256":records["KOSTYA"]["sha256"],
            "generated_images":records,
            "archive_sha256":hashlib.sha256((self.root/ARCHIVE).read_bytes()).hexdigest(),
            "contact_sha256":hashlib.sha256((self.root/CONTACT).read_bytes()).hexdigest()
        }
        (self.root/RECEIPT).write_text(json.dumps(data))
        return data

    def test_preserves_good_Kostya_without_generating_another(self):
        original,sha=original_kostya(self.root,self.nonce)
        self.assertEqual(original,self.raws["KOSTYA"])
        self.assertEqual(sha,hashlib.sha256(original).hexdigest())

    def test_prepared_worker_embeds_original_only_into_private_kernel(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"tools").mkdir()
            (root/"tools"/"kaggle_eb01_elizabeth_repair_entry.py").write_text(
                'CONFIG_B64="__EB01_ELIZABETH_REPAIR_CONFIG_B64__"\n')
            result=prepare("wolf19loner92","eb01repair-test-1",self.root,
                           self.nonce,root/"private_job",root=root)
            self.assertEqual(result["new_images"],1)
            self.assertEqual(result["reused_good_existing_images"],1)
            self.assertFalse(result["gpu_executed"])
            self.assertEqual(result["original_KOSTYA_sha256"],
                             hashlib.sha256(self.raws["KOSTYA"]).hexdigest())
            self.assertNotIn("KAGGLE_API_TOKEN",(root/"private_job"/"kernel_main.py").read_text())

    def test_mixed_repair_keeps_Kostya_hash(self):
        self.save(previous=False)
        sha=hashlib.sha256(self.raws["KOSTYA"]).hexdigest()
        v=verify(self.root,self.nonce,sha)
        self.assertEqual(v["status"],"PHOTOREAL_ELIZABETH_REPAIRED_KOSTYA_PRESERVED")
        self.assertEqual(v["source_model_ELIZABETH"],PHOTO_MODEL)
        self.assertFalse(v["visual_identity_approved"])

    def test_wrong_claim_of_reusing_Kostya_fails(self):
        data=self.save(previous=False)
        data["generated_images"]["KOSTYA"]["reused_unchanged"]=False
        (self.root/RECEIPT).write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"Mixed canon"):
            verify_canon(self.root,self.nonce)


if __name__=="__main__":
    unittest.main()
