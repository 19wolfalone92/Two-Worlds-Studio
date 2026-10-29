"""Regression coverage for ONE Studio animation pipeline (no private art or GPU)."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw
from studio.animation_pipeline import prepare_auto_rig, two_bone_ik, walk_tracks
from studio.character_engine import CharacterRig
from studio.cinematic_lighting import LightStage, validate_lights
from studio.facial_actor import PerformanceRig
from studio.deformable_actor import DeformableActor, validate_deform
from studio.cinematic import CinematicPanel

try:
    import cv2
except ImportError:
    cv2=None


class AnimationProductionV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.size=(320,480)
        im=Image.new('RGBA',self.size,(0,0,0,0))
        dr=ImageDraw.Draw(im)
        dr.ellipse((133,54,187,134),fill=(185,70,125,255))
        dr.polygon([(121,125),(199,125),(214,275),(116,275)],fill=(50,100,200,255))
        dr.polygon([(121,139),(104,139),(95,320),(122,320)],fill=(190,95,85,255))
        dr.polygon([(199,139),(216,139),(226,320),(198,320)],fill=(185,90,75,255))
        dr.polygon([(124,270),(158,270),(158,430),(125,430)],fill=(90,90,95,255))
        dr.polygon([(163,270),(197,270),(197,430),(162,430)],fill=(90,90,95,255))
        dr.rectangle((111,428,160,449),fill=(40,40,45,255))
        dr.rectangle((159,428,214,449),fill=(40,40,45,255))
        self.image=self.root/'actor.png'
        im.save(self.image)

    def tearDown(self):
        self.tmp.cleanup()

    def _rig(self):
        cfg=prepare_auto_rig(self.image,self.root/'rig',
                             body_id='earth_mira',camera_view='back')
        return cfg,CharacterRig(cfg,self.root/'rig',self.size)

    def test_auto_rig_creates_real_parts_and_moves(self):
        cfg,rig=self._rig()
        self.assertEqual(len(cfg['parts']),14)
        self.assertFalse(cfg['owner_approved'])
        self.assertFalse(cfg['autorig_anatomy_verified'])
        self.assertEqual(cfg['camera_view'],'back')
        self.assertNotEqual(rig.frame(0).tobytes(),rig.frame(1).tobytes())
        from studio.mirror_rear_rig import validate_mira_rear_contract
        validate_mira_rear_contract(cfg)
        for part in cfg['parts']:
            self.assertEqual(hashlib.sha256(
                (self.root/'rig'/part['path']).read_bytes()).hexdigest(),part['sha256'])
        with self.assertRaises(FileExistsError):
            prepare_auto_rig(self.image,self.root/'rig',body_id='earth_mira')

    def test_ik_geometry_and_failure(self):
        for bend in (-1,1):
            knee,upper,lower=two_bone_ik((0,0),(15,80),50,50,bend)
            self.assertAlmostEqual(math.dist((0,0),knee),50,places=6)
            self.assertAlmostEqual(math.dist(knee,(15,80)),50,places=6)
            self.assertTrue(math.isfinite(upper))
            self.assertTrue(math.isfinite(lower))
        for target in [(0,200),(0,0)]:
            with self.assertRaises(ValueError):
                two_bone_ik((0,0),target,50,50)

    def test_back_art_cannot_be_derived_from_matte_jpeg(self):
        Image.open(self.image).convert('RGB').save(self.root/'matte.jpg')
        with self.assertRaisesRegex(ValueError,'transparent'):
            prepare_auto_rig(self.root/'matte.jpg',self.root/'bad',body_id='earth_mira')

    def test_light_pass_is_deterministic_and_validated(self):
        spec=[{'x':.5,'y':.5,'radius':.6,'rgb':[80,140,255],'from':0,'to':.4}]
        light=LightStage(spec,self.size)
        original=Image.new('RGB',self.size,(35,35,40))
        self.assertEqual(light.frame(original,0).tobytes(),original.tobytes())
        self.assertEqual(light.frame(original,.7).tobytes(),light.frame(original,.7).tobytes())
        self.assertNotEqual(light.frame(original,.8).tobytes(),original.tobytes())
        with self.assertRaises(ValueError):
            validate_lights([{'x':float('nan'),'y':.3,'radius':.8,'rgb':[1,1,1]}])

    def test_facial_plate_states_and_back_view_protection(self):
        cfg,rig=self._rig()
        with self.assertRaisesRegex(ValueError,'BACK'):
            PerformanceRig(rig,{'eyes':{}},self.root,duration=3)
        front=copy.deepcopy(cfg)
        front['camera_view']='front'
        actor=CharacterRig(front,self.root/'rig',self.size)
        facial={'eyes':{},'mouth':{},'blinks':[.1],
                'visemes':[{'at':0,'shape':'rest'},{'at':.5,'shape':'wide'}]}
        for category,keys in [('eyes',('open','half','closed')),
                              ('mouth',('rest','small','wide'))]:
            for i,key in enumerate(keys):
                plate=Image.new('RGBA',self.size)
                ImageDraw.Draw(plate).rectangle(
                    (132,85 if category=='eyes' else 103,
                     183,103 if category=='eyes' else 119),
                    fill=(20+i*90,30,30,255))
                path=self.root/f'{category}_{key}.png'
                plate.save(path)
                facial[category][key]={
                    'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        perf=PerformanceRig(actor,facial,self.root,duration=1)
        self.assertEqual(perf.mode,'timed_visemes')
        self.assertNotEqual(perf.frame(.11).tobytes(),perf.frame(.8).tobytes())
        facial['mouth']['wide']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA-256'):
            PerformanceRig(actor,facial,self.root,duration=1)

    @unittest.skipUnless(cv2 is not None,'Optional OpenCV not installed')
    def test_continuous_deformation_changes_actual_pixels(self):
        spec={'schema':1,'controls':[
            {'name':'head','x':.5,'y':.23,'radius':.32,
             'keys':[{'at':0,'dx':0,'dy':0},{'at':1,'dx':.02,'dy':-.02}]},
            {'name':'hip','x':.5,'y':.66,'radius':.38,
             'keys':[{'at':0,'dx':0,'dy':0},{'at':1,'dx':0,'dy':0}]}]}
        moving=DeformableActor(Image.open(self.image),spec)
        self.assertNotEqual(moving.frame(0).tobytes(),moving.frame(1).tobytes())
        self.assertEqual(moving.frame(.5).tobytes(),moving.frame(.5).tobytes())
        bad=copy.deepcopy(spec);bad['controls'][0]['keys'][1]['dx']=.4
        with self.assertRaisesRegex(ValueError,'safe movement'):
            validate_deform(bad)

    @unittest.skipUnless(cv2 is not None,'Optional OpenCV not installed')
    def test_auto_matte_converts_flat_gpu_background_to_unreviewed_rgba(self):
        from studio.auto_matte import matte_plain_background
        with Image.open(self.image) as im:
            rgba=im.convert('RGBA')
        flat=Image.new('RGBA',self.size,(211,210,208,255))
        flat.alpha_composite(rgba)
        flat.convert('RGB').save(self.root/'factory_rgb.png')
        result=matte_plain_background(self.root/'factory_rgb.png',self.root/'matte.png')
        self.assertFalse(result['owner_approved'])
        with Image.open(self.root/'matte.png') as matted:
            self.assertEqual(matted.mode,'RGBA')
            self.assertEqual(matted.getpixel((0,0))[3],0)
        with self.assertRaises(FileExistsError):
            matte_plain_background(self.root/'factory_rgb.png',self.root/'matte.png')

    @unittest.skipUnless(cv2 is not None,'Optional OpenCV not installed')
    def test_cinematic_integration_with_mesh_and_lights(self):
        scene={'id':'S01','duration':2,
               'cinematic':{'grain':0,'vignette':0,
               'lights':[{'x':.5,'y':.3,'radius':.8,'rgb':[135,90,255],
                          'from':0,'to':.4}],
               'layers':[{'role':'background','path':'background.png'},
                         {'role':'character','path':'actor.png',
                          'deform':{'schema':1,'controls':[
                          {'name':'head','x':.5,'y':.25,'radius':.3,
                           'keys':[{'at':0,'dx':0,'dy':0},
                                   {'at':1,'dx':.02,'dy':0}]},
                          {'name':'root','x':.5,'y':.6,'radius':.3,
                           'keys':[{'at':0,'dx':0,'dy':0},
                                   {'at':1,'dx':0,'dy':0}]}]}}],
               'camera':{'start':{'zoom':1},'end':{'zoom':1.02}}}}
        Image.new('RGB',self.size,(25,30,45)).save(self.root/'background.png')
        panel=CinematicPanel(scene,self.root,*self.size)
        self.assertNotEqual(panel.frame(0).tobytes(),panel.frame(.7).tobytes())


if __name__=='__main__':
    unittest.main()
