"""Work-only script lock, character body identity, audio provenance and visual direction."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from PIL import Image
from studio.identity_qc import verify_body, verify_episode_characters,load_registry
from studio.audio_master import audit_audio, mix_audio
from studio.work_contract import script_fingerprint,verify_work_lock
from studio.visual_director import planned_shots,classify
from studio.video_qc import sample_video
from studio.render import render
from studio.premium_assets import inspect_image

ROOT=Path(__file__).resolve().parent.parent
class StudioHardeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.assets=self.root/"assets"
        self.assets.mkdir()
        self.e1=json.loads((ROOT/"episodes/E01_THE_REFLECTION.json").read_text(encoding="utf-8"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_work_lock_changes_detected(self):
        ep=copy.deepcopy(self.e1)
        fp=script_fingerprint(ep)
        ep["work_lock"]={"origin":"chatgpt_work","script_sha256":fp,
                         "editorial_approved":True,"approved_by":"owner"}
        self.assertTrue(verify_work_lock(ep)["work_lock_valid"])
        ep["scenes"][0]["caption"]="Unexpected rewrite of Work dialogue!"
        with self.assertRaisesRegex(ValueError,"changed after Work lock"):
            verify_work_lock(ep)

    def test_owner_approval_cannot_be_assumed(self):
        ep=copy.deepcopy(self.e1)
        ep["work_lock"]={"origin":"chatgpt_work","script_sha256":script_fingerprint(ep),
                         "editorial_approved":False}
        with self.assertRaisesRegex(ValueError,"not explicitly editorially approved"):
            verify_work_lock(ep)

    def test_story_never_generated_by_visual_director(self):
        before=copy.deepcopy(self.e1)
        plan=planned_shots(self.e1)
        self.assertEqual(plan["episode"],"E01")
        self.assertEqual(len(plan["shots"]),len(self.e1["scenes"]))
        self.assertEqual(self.e1,before)
        self.assertEqual(plan["script_sha256"],script_fingerprint(self.e1))
        self.assertFalse(any("dialogue" in s for s in plan["shots"]))

    def test_director_requires_work_lock_for_production(self):
        with self.assertRaisesRegex(ValueError,"No Work-authored"):
            planned_shots(self.e1,require_work_lock=True)

    def test_missing_references_block_premium_identity(self):
        registry=load_registry()
        with self.assertRaisesRegex(ValueError,"approved front/three-quarter/profile"):
            verify_body("earth_mira","MIRA_EARTH_B01_v001",registry,self.assets,strict=True)
        with self.assertRaisesRegex(ValueError,"version drift"):
            verify_body("earth_mira","wrong",registry,self.assets,strict=True)

    def test_valid_registered_reference_views(self):
        ref=[]
        for i,name in enumerate(("front","three_quarter","profile")):
            pic=Image.new("RGB",(640,704),(110+10*i,70,95))
            path=self.assets/f"{name}.png"
            pic.save(path)
            ref.append({"view":name,"file":path.name,
                        "sha256":inspect_image(path)["sha256"],
                        "review_status":"approved","rights":"owned",
                        "rights_note":"Artist-produced original owned concept art"})
        reg={"bodies":{"earth_mira":{"version":"B01_v1","references":ref}}}
        result=verify_body("earth_mira","B01_v1",reg,self.assets,strict=True)
        self.assertFalse(result["visual_match_verified"])
        self.assertEqual(result["checked"],3)
        wrong=copy.deepcopy(reg)
        wrong["bodies"]["earth_mira"]["references"][2]["sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"Reference changed"):
            verify_body("earth_mira","B01_v1",wrong,self.assets,strict=True)

    def _record_wav(self,name="voice.wav",seconds=.65):
        sr=22050
        n=int(sr*seconds)
        samples=(np.sin(np.arange(n,dtype=np.float64)*2*math.pi*180/sr)*9000).astype(np.int16)
        path=self.assets/name
        with wave.open(str(path),"wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(sr)
            stream.writeframes(samples.tobytes())
        return path

    @unittest.skipUnless(shutil.which("ffprobe"),"ffprobe required")
    def test_voice_profile_and_scene_timing(self):
        rec=self._record_wav()
        regfile=self.root/"voices.json"
        regfile.write_text(json.dumps({"schema":1,"speakers":{
            "Mira":{"version":"v1","review_status":"approved"},
            "Lyra":{"version":"v2","review_status":"approved"}}}),encoding="utf-8")
        ep={"id":"E77","scenes":[{"id":"S01","duration":1.0,"voiceover":"Hello.",
           "continuity":{"state":"post","body":"noctaris_lyra","occupant":"Mira"},
           "voice":{"speaker":"Mira","voice_version":"v1","path":"voice.wav",
                    "review_status":"approved","rights":"owned",
                    "rights_note":"Recorded by the actor with commercial rights",
                    "sha256":hashlib.sha256(rec.read_bytes()).hexdigest()}}]}
        ok=audit_audio(ep,self.assets,publish=True,registry_path=regfile,fps=24)
        self.assertEqual(len(ok["tracks"]),1)
        bad=copy.deepcopy(ep)
        bad["scenes"][0]["voice"]["speaker"]="Lyra"
        bad["scenes"][0]["voice"]["voice_version"]="v2"
        with self.assertRaisesRegex(ValueError,"not in scene"):
            audit_audio(bad,self.assets,publish=True,registry_path=regfile)
        bad=copy.deepcopy(ep)
        bad["scenes"][0]["voice"]["sha256"]="deadbeef"
        with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
            audit_audio(bad,self.assets,publish=True,registry_path=regfile)

    @unittest.skipUnless(shutil.which("ffprobe"),"ffprobe required")
    def test_dialogue_without_recording_not_master_ready(self):
        ep={"scenes":[{"id":"S01","duration":2.0,"voiceover":"Hello.",
                      "continuity":{"state":"post","body":"earth_mira","occupant":"Lyra"}}]}
        with self.assertRaisesRegex(ValueError,"missing approved voice"):
            audit_audio(ep,self.assets,publish=True)

    @unittest.skipUnless(shutil.which("ffprobe") and shutil.which("ffmpeg"),
                         "FFmpeg required")
    def test_real_audio_mix_is_not_synthetic_voice(self):
        audio=self._record_wav(seconds=.6)
        scene={"id":"S01","duration":1.0,"palette":"earth",
               "effect":"none","heading":"HELLO","caption":"Hello."}
        epfile=self.root/"scene.json"
        epfile.write_text(json.dumps({"id":"E88","scenes":[scene]}),encoding="utf-8")
        video=self.root/"raw.mp4"
        render(epfile,video,self.assets,180,320,6)
        mixed=self.root/"mixed.mp4"
        out=mix_audio(video,[{"kind":"voice","path":str(audio),"start_ms":150}],output=mixed)
        self.assertTrue(out.is_file())
        self.assertGreater(out.stat().st_size,10000)

    def test_approved_reference_motion_requires_rights(self):
        from studio.actor_motion import validate_motion,animate_actor
        self.assertEqual(validate_motion({"breath_scale":.02,"sway_x":.01},
                                          self.assets)["blink_frames"],0)
        with self.assertRaisesRegex(ValueError,"breath_scale"):
            validate_motion({"breath_scale":.3},self.assets)
        with self.assertRaises(FileNotFoundError):
            validate_motion({"blink":{"path":"missing.png","at":[.4]}},self.assets)
        pic=Image.new("RGBA",(160,270),(0,0,0,0))
        from PIL import ImageDraw
        ImageDraw.Draw(pic).ellipse((40,55,95,110),fill=(245,220,200,255))
        start=animate_actor(pic,{"breath_scale":.02,"sway_x":.01},0)
        end=animate_actor(pic,{"breath_scale":.02,"sway_x":.01},.3)
        self.assertNotEqual(start.tobytes(),end.tobytes())

    @unittest.skipUnless(shutil.which("ffmpeg"),"ffmpeg required")
    def test_video_diagnostics_are_advisory(self):
        from studio.render import render
        epfile=self.root/"ep.json"
        epfile.write_text(json.dumps({"scenes":[{"id":"S01","duration":1.1,
                 "palette":"earth","effect":"rain",
                 "heading":"TEST","caption":"A test"}]}),encoding="utf-8")
        video=self.root/"test.mp4"
        render(epfile,video,self.assets,180,320,6)
        data=sample_video(video,fps=2)
        self.assertGreater(data["frames_sampled"],0)
        self.assertFalse(data["automated_aesthetic_approval"])

if __name__=="__main__":
    unittest.main()
