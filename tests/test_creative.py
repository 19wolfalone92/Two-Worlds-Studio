"""Art prompting tests only. This studio never writes story scripts."""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
from studio.art_director import ROOT,load_style,compile_prompt_pack,panel_prompt,comfy_workflow,run

class StudioArtDirectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.style=load_style()
        cls.e1=json.loads((ROOT/"episodes/E01_THE_REFLECTION.json").read_text(encoding="utf-8"))

    def test_stable_doubles(self):
        a=self.style["characters"]
        self.assertEqual(a["earth_mira"]["identity_id"],a["noctaris_lyra"]["identity_id"])

    def test_reference_later_not_claimed_verified(self):
        pack=compile_prompt_pack(self.e1,self.style)
        self.assertEqual(len(pack["scenes"]),6)
        self.assertEqual(pack["backend"],"unexecuted")
        self.assertEqual(pack,compile_prompt_pack(self.e1,self.style))
        self.assertIn("earth_mira",pack["scenes"][3]["bodies"])
        self.assertIn("noctaris_lyra",pack["scenes"][3]["bodies"])
        self.assertTrue(all(not s["consistency_verified"] for s in pack["scenes"]))

    def test_prompt_only_backend_never_renders_image(self):
        with tempfile.TemporaryDirectory() as temp:
            file=run(ROOT/"episodes/E01_THE_REFLECTION.json",Path(temp))
            self.assertTrue(file.exists())
            self.assertFalse(any(Path(temp).rglob("*.png")))

    def test_ref_missing_refused(self):
        bad=copy.deepcopy(self.style)
        bad["characters"]["earth_mira"]["reference_file"]="not_installed.png"
        with self.assertRaisesRegex(ValueError,"Missing or unsafe"):
            panel_prompt(self.e1["scenes"][0],bad,"E01")

    def test_comfy_workflow_uses_correct_typed_values(self):
        template=json.loads((ROOT/"templates/comfyui_sdxl_api.json").read_text(encoding="utf-8"))
        prompt=panel_prompt(self.e1["scenes"][0],self.style,"E01")
        result=comfy_workflow(prompt,template,"licensed.safetensors",768,1344)
        self.assertEqual(result["5"]["inputs"]["width"],768)
        self.assertEqual(result["3"]["inputs"]["seed"],prompt["seed"])

if __name__=="__main__":
    unittest.main()
