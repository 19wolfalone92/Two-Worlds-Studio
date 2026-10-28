"""Functional tests for parented sprite transforms and scene compositing."""
from __future__ import annotations
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw
from studio.character_engine import CharacterRig, validate_rig
from studio.cinematic import CinematicPanel
from studio.animation_quality import audit_performance

class CharacterAnimationEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.w,self.h=128,224
        def sprite(name, shape, fill):
            im=Image.new("RGBA",(self.w,self.h),(0,0,0,0))
            dr=ImageDraw.Draw(im)
            dr.ellipse(shape,fill=fill)
            p=self.root/f"{name}.png";im.save(p)
            return {"name":name,"path":p.name}
        self.body=sprite("body",(45,90,83,176),(120,75,160,255))
        self.head=sprite("head",(46,49,80,94),(230,169,144,255))
        self.hand=sprite("hand",(70,97,88,147),(205,160,125,255))
        self.rig={"schema":1,"root_pivot":[.5,.72],
          "parts":[dict(self.body,pivot=[.5,.52],z=1),
                   dict(self.head,pivot=[.5,.41],parent="body",z=3),
                   dict(self.hand,pivot=[.58,.49],parent="body",z=2)],
          "tracks":{
             "__root__":[{"at":0,"x":-.08},{"at":1,"x":.10}],
             "head":[{"at":0,"angle":-12},{"at":1,"angle":12}],
             "hand":[{"at":0,"angle":5},{"at":1,"angle":-65}]
          },
          "ground_shadow":{"center":[.5,.85],"radii":[.2,.033],"opacity":.3,"blur":.01},
          "environment_light":{"rgb":[75,94,165],"strength":.10}
        }
        bg=Image.new("RGB",(self.w,self.h),(23,30,49))
        ImageDraw.Draw(bg).rectangle((0,176,self.w,self.h),fill=(47,51,70))
        bg.save(self.root/"room.png")
        blank=Image.new("RGBA",(self.w,self.h),(0,0,0,0));blank.save(self.root/"character.png")

    def tearDown(self):
        self.tmp.cleanup()

    def test_three_part_hierarchy_moves_each_joint(self):
        stats=validate_rig(self.rig,self.root,(self.w,self.h))
        self.assertTrue(stats["is_acting"])
        character=CharacterRig(self.rig,self.root,(self.w,self.h))
        first=character.frame(0)
        middle=character.frame(.5)
        end=character.frame(1)
        self.assertNotEqual(hashlib.sha256(first.tobytes()).digest(),
                            hashlib.sha256(end.tobytes()).digest())
        self.assertNotEqual(first.tobytes(),middle.tobytes())
        self.assertEqual(first.size,(self.w,self.h))
        self.assertIsNotNone(first.getchannel("A").getbbox())
        self.assertEqual(character.frame(.5).tobytes(),middle.tobytes())

    def test_parent_rotation_affects_child_joint(self):
        rig=copy.deepcopy(self.rig)
        rig["tracks"]={"body":[{"at":0,"angle":0},{"at":1,"angle":40}]}
        animator=CharacterRig(rig,self.root,(self.w,self.h))
        start=animator.pose(0)["head"]
        end=animator.pose(1)["head"]
        self.assertNotEqual(round(float(start[0,2]),5),round(float(end[0,2]),5))

    def test_invalid_rigs_fail_before_production(self):
        bad=copy.deepcopy(self.rig)
        bad["parts"][1]["parent"]="unknown"
        with self.assertRaisesRegex(ValueError,"unknown parent"):
            validate_rig(bad,self.root,(self.w,self.h))
        bad=copy.deepcopy(self.rig)
        bad["parts"][0]["parent"]="head"
        with self.assertRaisesRegex(ValueError,"cyclic"):
            validate_rig(bad,self.root,(self.w,self.h))
        bad=copy.deepcopy(self.rig)
        bad["tracks"]["head"]=[{"at":.8},{"at":.3}]
        with self.assertRaisesRegex(ValueError,"unsorted"):
            validate_rig(bad,self.root,(self.w,self.h))
        bad=copy.deepcopy(self.rig)
        bad["tracks"]["head"]=[{"at":0,"angle":200}]
        with self.assertRaisesRegex(ValueError,"angle"):
            validate_rig(bad,self.root,(self.w,self.h))

    def test_master_requires_sha_approval_on_each_sprite(self):
        with self.assertRaisesRegex(ValueError,"unapproved rig artwork"):
            validate_rig(self.rig,self.root,(self.w,self.h),publish=True)
        complete=copy.deepcopy(self.rig)
        for part in complete["parts"]:
            part["sha256"]=hashlib.sha256((self.root/part["path"]).read_bytes()).hexdigest()
            part["review_status"]="approved"
            part["rights"]="original"
            part["rights_note"]="Artist-produced original visual rig component."
        self.assertTrue(validate_rig(complete,self.root,(self.w,self.h),publish=True)["is_acting"])

    def test_actual_rig_renders_through_existing_cinematic_pipeline(self):
        scene={"id":"S01","duration":2.0,
            "cinematic":{"layers":[
                {"role":"background","path":"room.png","depth":-.8},
                {"role":"character","path":"character.png","depth":.4,"rig":self.rig}],
                "camera":{"start":{"zoom":1},"end":{"zoom":1.06}},
                "grain":0}}
        img1=CinematicPanel(scene,self.root,self.w,self.h).frame(0)
        img2=CinematicPanel(scene,self.root,self.w,self.h).frame(1)
        self.assertEqual(img1.size,(self.w,self.h))
        self.assertNotEqual(img1.tobytes(),img2.tobytes())

    def test_master_motion_qc_requires_real_rig(self):
        scene={"id":"S01","cinematic":{"layers":[
            {"role":"background"},{"role":"character","rig":self.rig}]},
            "animation":{"kind":"rigged_2d","cues":["pose","hand_action"],
                "integration":{"contact_shadow":True,"lighting":True,"perspective":True},
                "reviewed_by":"Art reviewer"}}
        self.assertTrue(audit_performance({"scenes":[scene]},publish=True)["master_ready"])
        incomplete=copy.deepcopy(scene)
        del incomplete["cinematic"]["layers"][1]["rig"]
        with self.assertRaisesRegex(ValueError,"actual articulated"):
            audit_performance({"scenes":[incomplete]},publish=True)

if __name__=="__main__":
    unittest.main()
