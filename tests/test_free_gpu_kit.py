"""No GPU/model downloads needed to validate the free Kaggle/voice handoff."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from studio.art_director import ROOT,load_style,compile_prompt_pack
from studio.kaggle_flux import character_job,scene_job,plan_job,execute,validate_reference
from studio.kokoro_tts import jobs_from_episode,load_cast,render_jobs

class KaggleFreeKitTests(unittest.TestCase):
    def setUp(self):
        self.style=load_style()
        self.cast=load_cast()
    def test_character_dry_run_creates_no_image(self):
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(mode="character",body="earth_mira",view="front",
                 reference=None,prompt_pack=None,scene=None,output_dir=d,
                 dry_run=True,width=576,height=1024,steps=4)
            planned=execute(args)
            self.assertTrue(planned["dry_run"])
            self.assertEqual(planned["real_images_generated"],0)
            self.assertEqual(list(Path(d).iterdir()),[])
            self.assertEqual(planned["job"]["body"],"earth_mira")
            self.assertFalse(planned["job"]["visual_identity_verified"])
    def test_turnarounds_share_identity_seed(self):
        first=character_job(self.style,"noctaris_lyra","front")
        second=character_job(self.style,"noctaris_lyra","profile")
        self.assertNotEqual(first["seed"],second["seed"])
        self.assertIn("side profile",second["prompt"].lower())
        self.assertEqual(first["review_status"],"unreviewed")
    def test_reference_file_hash(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"portrait.png"
            Image.new("RGB",(256,256),(160,112,108)).save(path)
            detail=validate_reference(path)
            self.assertEqual(len(detail["sha256"]),64)
    def test_unknown_character_rejected(self):
        with self.assertRaisesRegex(ValueError,"Unknown physical"):
            character_job(self.style,"no_such_person","front")
    def test_scene_prompt_pack_can_be_imported_without_gpu(self):
        ep=json.loads((ROOT/"episodes/E01_THE_REFLECTION.json").read_text(encoding="utf-8"))
        pack=compile_prompt_pack(ep,self.style)
        job=scene_job(pack,"S01")
        self.assertEqual(job["episode"],"E01")
        self.assertIn("cinematic",job["prompt"].lower())
        with self.assertRaisesRegex(ValueError,"not found"):
            scene_job(pack,"S97")
    def test_cast_has_distinct_stable_voice_profiles(self):
        speakers=self.cast["speakers"]
        self.assertNotEqual(speakers["Mira"]["voice"],speakers["Lyra"]["voice"])
        self.assertEqual(speakers["Lyra"]["identity_version"],"LYRA_VOICE_v001")
    def test_tts_dry_run_generates_no_sound(self):
        with tempfile.TemporaryDirectory() as d:
            data=render_jobs([{"scene":"SAMPLE","speaker":"Mira",
                    "text":"Voice check only.","voice":"af_heart","language":"a"}],
                    Path(d),dry_run=True)
            self.assertEqual(data["actual_audio_clips"],0)
            self.assertEqual(list(Path(d).iterdir()),[])
    def test_no_speaker_inference_from_script(self):
        episode={"scenes":[{"id":"S01","duration":2.3,
                     "continuity":{"state":"post","body":"noctaris_lyra",
                                   "occupant":"Mira"},
                     "voiceover":"I cannot speak Noctari."}]}
        self.assertEqual(jobs_from_episode(episode,self.cast),[])
    def test_tts_requires_correct_acting_consciousness(self):
        ep={"scenes":[{"id":"S01","duration":2.3,"tts":{"speaker":"Mira","text":"Voice test."},
                      "continuity":{"state":"post","body":"noctaris_lyra",
                                    "occupant":"Mira"}}]}
        self.assertEqual(jobs_from_episode(ep,self.cast)[0]["speaker"],"Mira")
        wrong=copy.deepcopy(ep)
        wrong["scenes"][0]["tts"]["speaker"]="Lyra"
        with self.assertRaisesRegex(ValueError,"not represented"):
            jobs_from_episode(wrong,self.cast)
    def test_kaggle_notebook_is_valid_self_contained(self):
        notebook=json.loads((ROOT/"notebooks/TWO_WORLDS_Kaggle_FLUX_Kokoro.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"],4)
        self.assertGreaterEqual(len(notebook["cells"]),10)
        code="".join("".join(cell.get("source",[])) for cell in notebook["cells"])
        self.assertIn("ENABLE_GPU_IMAGE = False",code)
        self.assertIn("ENABLE_KOKORO_VOICE = False",code)
        self.assertNotIn("ghp_",code)
if __name__=="__main__":
    unittest.main()
