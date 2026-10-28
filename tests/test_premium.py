"""No screenplay creation: visual quality gates and deterministic 2.5D animation."""
from __future__ import annotations
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw

from studio.premium_assets import (resolve_asset, inspect_image, validate_layer,
                                   validate_premium_scene, review_episode)
from studio.cinematic import CinematicPanel, camera_state, ease
from studio.review_board import contact_sheet
from studio.render import render, validate_episode
from studio.visual_prep import make_demo_assets

class PremiumVisualTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.assets=self.root/"assets"
        (self.assets/"demo").mkdir(parents=True,exist_ok=True)
        bg=Image.new("RGB",(320,576),(30,35,76))
        bd=ImageDraw.Draw(bg)
        bd.polygon([(0,400),(220,180),(320,400),(320,576),(0,576)],fill=(175,95,115))
        bg.save(self.assets/"demo"/"background.png")
        fg=Image.new("RGBA",(320,576),(0,0,0,0))
        fd=ImageDraw.Draw(fg)
        fd.ellipse((140,170,260,280),fill=(241,192,163,230))
        fg.save(self.assets/"demo"/"subject.png")
        self.scene={"id":"S01","duration":1.0,"palette":"noctaris",
                    "effect":"none","heading":"TEST","caption":"Visual proof",
                    "cinematic":{"camera":{"start":{"x":-.25,"y":0,"zoom":1.02},
                                            "end":{"x":.25,"y":.2,"zoom":1.20}},
                                 "grain":0.03,"vignette":0.20,"layers":[
                      {"role":"background","path":"demo/background.png","depth":-.8},
                      {"role":"character","path":"demo/subject.png","depth":.8}]}}
        self.episode={"id":"E99","scenes":[self.scene]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_secure_paths(self):
        self.assertTrue(resolve_asset(self.assets,"demo/background.png").is_file())
        for s in ("/etc/passwd","../escape.png","demo/../../wrong.png","demo/text.txt"):
            with self.assertRaises(ValueError):
                resolve_asset(self.assets,s)

    def test_detect_modified_image_hash(self):
        valid=copy.deepcopy(self.scene)
        valid["cinematic"]["layers"][0]["sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"hash mismatch"):
            validate_premium_scene(valid,self.assets)

    def test_preflight_reports_warnings_not_artistic_certification(self):
        report=review_episode(self.episode,self.assets,publish=False,min_width=256,min_height=448)
        self.assertEqual(report["cinematic_panels"],1)
        self.assertIn("technical_pass",report)
        self.assertFalse(any("identity" in str(x).lower() for x in report["results"]))

    def test_publish_gate_requires_rights_and_human_approval(self):
        with self.assertRaisesRegex(ValueError,"not explicitly approved"):
            review_episode(self.episode,self.assets,publish=True,min_width=256,min_height=448)
        approved=copy.deepcopy(self.episode)
        for layer in approved["scenes"][0]["cinematic"]["layers"]:
            layer["review_status"]="approved"
            layer["rights"]="original"
            layer["rights_note"]="Original artwork created for TW."
        report=review_episode(approved,self.assets,publish=True,min_width=256,min_height=448)
        self.assertTrue(report["technical_pass"])

    def test_production_quality_requires_resolution(self):
        approved=copy.deepcopy(self.episode)
        for layer in approved["scenes"][0]["cinematic"]["layers"]:
            layer.update(review_status="approved",rights="owned",rights_note="Original art with full rights")
        with self.assertRaisesRegex(ValueError,"resolution"):
            review_episode(approved,self.assets,publish=True,min_width=1080,min_height=1920)

    def test_camera_eases_and_parallax_changes_frames(self):
        self.assertEqual(ease(-5),0)
        self.assertEqual(ease(5),1)
        self.assertGreater(camera_state(self.scene,.7)[2],camera_state(self.scene,.1)[2])
        panel=CinematicPanel(self.scene,self.assets,180,320)
        a=panel.frame(0)
        b=panel.frame(1)
        self.assertEqual(a.size,(180,320))
        self.assertNotEqual(a.tobytes(),b.tobytes())

    def test_review_board_has_honest_labels(self):
        destination=self.root/"contact.png"
        result=contact_sheet(self.episode,self.assets,destination,
                             thumb_width=180,thumb_height=320)
        self.assertTrue(destination.is_file())
        self.assertFalse(result["visual_approval"])
        self.assertEqual(result["scenes"][0]["preview_mode"],"LAYERED PREVIEW")

    def test_render_path_supports_actual_layers(self):
        import shutil
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not installed")
        epfile=self.root/"scene.json"
        epfile.write_text(json.dumps(self.episode),encoding="utf-8")
        result=render(epfile,self.root/"preview.mp4",self.assets,180,320,6)
        self.assertGreater(result["frames"],0)
        self.assertTrue((self.root/"preview.mp4").stat().st_size>10000)

    def test_demo_assets_do_not_mark_art_approved(self):
        folder=self.root/"procedural"
        spec,ep=make_demo_assets(folder,width=360,height=640)
        self.assertTrue(spec.is_file())
        self.assertEqual(len(ep["scenes"][0]["cinematic"]["layers"]),3)
        with self.assertRaisesRegex(ValueError,"not explicitly approved"):
            review_episode(ep,folder/"assets",publish=True,min_width=256,min_height=448)

    def test_scene_camera_bad_zoom_fails(self):
        scene=copy.deepcopy(self.scene)
        scene["cinematic"]["camera"]["end"]["zoom"]=20
        with self.assertRaisesRegex(ValueError,"Camera zoom"):
            validate_premium_scene(scene,self.assets)

    def test_procedural_scenes_are_never_premium_release(self):
        with self.assertRaisesRegex(ValueError,"premium release"):
            review_episode({"id":"E01","scenes":[{"id":"S01"}]},self.assets,publish=True)

if __name__=="__main__":
    unittest.main()
