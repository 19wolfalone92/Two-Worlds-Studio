import copy
import json
import tempfile
import unittest
from pathlib import Path
from studio.art_director import ROOT, load_style, compile_prompt_pack, panel_prompt, comfy_workflow, run
from studio.story_writer import draft, read_json, PLAN, MEMORY, validate_plan, validate_dialogue
from studio.continuity import check_sequence

class StudioCreativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.style=load_style()
        cls.e1=read_json(ROOT/"episodes/E01_THE_REFLECTION.json")

    def test_identity_anchors_paired(self):
        a=self.style["characters"]
        self.assertEqual(a["earth_mira"]["identity_id"],a["noctaris_lyra"]["identity_id"])
        self.assertEqual(a["earth_mira"]["seed_anchor"],a["noctaris_lyra"]["seed_anchor"])

    def test_prompt_pack_stable_and_unapproved(self):
        one=compile_prompt_pack(self.e1,self.style)
        two=compile_prompt_pack(self.e1,self.style)
        self.assertEqual(one,two)
        self.assertEqual(len(one["scenes"]),6)
        self.assertIn("earth_mira",one["scenes"][3]["bodies"])
        self.assertIn("noctaris_lyra",one["scenes"][3]["bodies"])
        self.assertFalse(one["scenes"][0]["consistency_verified"])

    def test_prompts_not_images_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp=run(ROOT/"episodes/E01_THE_REFLECTION.json",Path(tmp))
            data=json.loads(fp.read_text(encoding="utf-8"))
            self.assertEqual(data["backend"],"unexecuted")
            self.assertFalse(any(Path(tmp).rglob("*.png")))

    def test_reference_missing_errors(self):
        bad=copy.deepcopy(self.style)
        bad["characters"]["earth_mira"]["reference_file"]="not_available.png"
        with self.assertRaisesRegex(ValueError,"Missing or unsafe"):
            panel_prompt(self.e1["scenes"][0],bad,"E01")

    def test_comfy_graph_valid_typed_values(self):
        graph=read_json(ROOT/"templates/comfyui_sdxl_api.json")
        s=panel_prompt(self.e1["scenes"][0],self.style,"E01")
        rendered=comfy_workflow(s,graph,"licensed.safetensors",768,1344)
        self.assertEqual(rendered["5"]["inputs"]["width"],768)
        self.assertIsInstance(rendered["3"]["inputs"]["seed"],int)
        with self.assertRaises(ValueError):
            comfy_workflow(s,graph,"licensed.safetensors",768,1345)

    def test_two_successive_drafts_keep_continuity(self):
        e3=draft("E03","outline")
        e4=draft("E04","outline")
        self.assertEqual(e3["editorial_status"],"draft_unapproved")
        self.assertTrue(check_sequence([e3,e4])["checked"])
        self.assertNotEqual(e3["title"],e4["title"])

    def test_mock_ollama_cannot_alter_plot_action(self):
        def fake(prompt,*,url,model):
            assert "CANON" in prompt
            beats=read_json(PLAN)["episodes"][0]["beats"]
            return {"scenes":[{"id":s["id"],"heading":s["heading"],
             "caption":s["caption"],"voiceover":s["voiceover"],
             "shot":"MAGIC TELEPORTS THE ENTIRE ARMY THROUGH THE MIRROR!"} for s in beats]}
        e3=draft("E03","ollama",generator=fake)
        first_beat=read_json(PLAN)["episodes"][0]["beats"][0]
        self.assertEqual(e3["scenes"][0]["shot"],first_beat["shot"])

    def test_prevent_engagement_bait(self):
        beats=read_json(PLAN)["episodes"][0]["beats"]
        cand={"scenes":[{"id":b["id"],"heading":b["heading"],
          "caption":"Vote now!", "voiceover":b["voiceover"],"shot":b["shot"]} for b in beats]}
        with self.assertRaisesRegex(ValueError,"prohibited"):
            validate_dialogue(cand,beats)

    def test_physical_world_binding(self):
        p=read_json(PLAN); m=read_json(MEMORY)
        p["episodes"][0]["beats"][0]["body"]="noctaris_lyra"
        with self.assertRaisesRegex(ValueError,"wrong physical world"):
            validate_plan(p,m)

    def test_unknown_draft_blocked(self):
        with self.assertRaisesRegex(ValueError,"not in season plan"):
            draft("E19","outline")

if __name__=="__main__": unittest.main()
