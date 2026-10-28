"""Work-produced image ingestion does not approve art or rewrite story."""
from __future__ import annotations
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw
from studio.asset_ingest import plan_ingestion, stage
from studio.work_contract import script_fingerprint

class WorkArtIngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.source=self.root/"input";self.source.mkdir()
        self.assets=self.root/"assets";self.assets.mkdir()
        img=Image.new("RGB",(640,960),(51,33,70))
        ImageDraw.Draw(img).rectangle((90,230,450,610),fill=(120,80,112))
        img.save(self.source/"bg.png")
        actor=Image.new("RGBA",(640,960),(0,0,0,0))
        ImageDraw.Draw(actor).ellipse((210,140,450,420),fill=(201,153,132,255))
        actor.save(self.source/"actor.png")
        self.ep={"id":"E77","title":"External Work story, not auto-written","scenes":[{
            "id":"S01","duration":2.0,"palette":"earth","effect":"none",
            "shot":"The character stands in front of an existing building entrance.",
            "caption":"A caption supplied by Work","voiceover":"",
            "continuity":{"state":"post","body":"earth_mira","occupant":"Lyra"}}],
            "continuity_start":"post","continuity_end":"post"}
        digest=script_fingerprint(self.ep)
        self.ep["work_lock"]={"origin":"chatgpt_work","script_sha256":digest,
                               "editorial_approved":True,"approved_by":"owner"}
        self.mp={"schema":1,"episode":"E77","script_sha256":digest,
                 "panels":{"S01":[
                     {"role":"background","source":"bg.png"},
                     {"role":"character","source":"actor.png","body_id":"earth_mira"}]}}
    def tearDown(self):
        self.tmp.cleanup()

    def test_staging_preserves_story_hash_and_copies_art(self):
        plan,transfers=plan_ingestion(self.ep,self.mp,self.source,self.assets)
        self.assertEqual(len(transfers),2)
        self.assertEqual(script_fingerprint(plan),script_fingerprint(self.ep))
        epfile=self.root/"script.json";epfile.write_text(json.dumps(self.ep),encoding="utf-8")
        mapfile=self.root/"map.json";mapfile.write_text(json.dumps(self.mp),encoding="utf-8")
        staged=self.root/"staged.json"
        result=stage(epfile,mapfile,self.source,self.assets,staged)
        self.assertEqual(result["staged_layers"],2)
        saved=json.loads(staged.read_text(encoding="utf-8"))
        self.assertEqual(saved["scenes"][0]["caption"],self.ep["scenes"][0]["caption"])
        self.assertEqual(saved["scenes"][0]["cinematic"]["layers"][1]["body_id"],"earth_mira")
        self.assertEqual(saved["scenes"][0]["cinematic"]["layers"][1]["review_status"],"unreviewed")
        for layer in saved["scenes"][0]["cinematic"]["layers"]:
            file=self.assets/layer["path"]
            self.assertTrue(file.is_file())
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),layer["sha256"])

    def test_wrong_character_identity_is_rejected(self):
        plan=copy.deepcopy(self.mp)
        plan["panels"]["S01"][1]["body_id"]="noctaris_lyra"
        with self.assertRaisesRegex(ValueError,"not listed"):
            plan_ingestion(self.ep,plan,self.source,self.assets)

    def test_mutated_script_revision_cannot_be_accepted(self):
        changed=copy.deepcopy(self.mp);changed["script_sha256"]="deadbeef"
        with self.assertRaisesRegex(ValueError,"different Work script"):
            plan_ingestion(self.ep,changed,self.source,self.assets)

    def test_external_relative_path_escape_is_rejected(self):
        changed=copy.deepcopy(self.mp)
        changed["panels"]["S01"][0]["source"]="../../secret.png"
        with self.assertRaisesRegex(ValueError,"unsafe"):
            plan_ingestion(self.ep,changed,self.source,self.assets)

    def test_unapproved_scenario_never_auto_stages(self):
        ep=copy.deepcopy(self.ep)
        ep["work_lock"]["editorial_approved"]=False
        with self.assertRaisesRegex(ValueError,"not explicitly editorially approved"):
            plan_ingestion(ep,self.mp,self.source,self.assets)

if __name__=="__main__":
    unittest.main()
