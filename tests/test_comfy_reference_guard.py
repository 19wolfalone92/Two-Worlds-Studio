"""ComfyUI IP-Adapter path tests: topology proof only, no GPU/model downloads."""
from __future__ import annotations
import copy
import json
import unittest

from studio.art_director import ROOT, comfy_workflow
from studio.comfy_reference_guard import audit_reference_conditioning


def graph():
    template = json.loads((ROOT / "templates/comfyui_sdxl_api.json").read_text(encoding="utf-8"))
    template["10"] = {"class_type": "LoadImage", "inputs": {"image": "__REFERENCE_IMAGE__"}}
    template["11"] = {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": "owner_reviewed_weights.bin"}}
    template["12"] = {
        "class_type": "IPAdapterAdvanced",
        "inputs": {"model": ["4", 0], "ipadapter": ["11", 0],
                   "image": ["10", 0], "weight": 0.7}
    }
    template["3"]["inputs"]["model"] = ["12", 0]
    return template


class ReferenceConditioningTests(unittest.TestCase):
    def test_reference_truly_feeds_saved_sampler_and_substitutes(self):
        template = graph()
        report = audit_reference_conditioning(template)
        self.assertTrue(report["topological_reference_path"])
        self.assertFalse(report["face_similarity_verified"])
        self.assertFalse(report["model_weights_verified"])
        prompt = {
            "prompt": "A woman at a window", "negative": "deformed",
            "seed": 17, "episode": "TEST01", "scene": "S01"
        }
        result = comfy_workflow(prompt, template, "sdxl.safetensors", 640, 1152,
                                reference_image="locked_portrait.png")
        self.assertEqual(result["10"]["inputs"]["image"], "locked_portrait.png")
        self.assertEqual(result["3"]["inputs"]["model"], ["12", 0])

    def test_unconnected_reference_does_not_trick_model(self):
        template = graph()
        template["3"]["inputs"]["model"] = ["4", 0]
        with self.assertRaisesRegex(ValueError, "does not use"):
            audit_reference_conditioning(template)

    def test_no_adapter_weight_and_no_adapter_model_are_blocked(self):
        template = graph()
        template["12"]["inputs"]["weight"] = 0
        with self.assertRaisesRegex(ValueError, "enabled IP-Adapter"):
            audit_reference_conditioning(template)
        template = graph()
        del template["12"]["inputs"]["ipadapter"]
        with self.assertRaisesRegex(ValueError, "enabled IP-Adapter"):
            audit_reference_conditioning(template)

    def test_load_image_not_connected_to_adapter_is_rejected(self):
        template = graph()
        template["13"] = {"class_type": "LoadImage", "inputs": {"image": "unrelated.png"}}
        template["12"]["inputs"]["image"] = ["13", 0]
        with self.assertRaisesRegex(ValueError, "enabled IP-Adapter"):
            audit_reference_conditioning(template)

    def test_no_reference_or_empty_graph_fails(self):
        template = json.loads((ROOT / "templates/comfyui_sdxl_api.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            audit_reference_conditioning(template)
        template = graph()
        del template["9"]
        with self.assertRaisesRegex(ValueError, "SaveImage node"):
            audit_reference_conditioning(template)

    def test_comfy_workflow_fails_before_submission_with_fake_reference_path(self):
        template = graph()
        template["3"]["inputs"]["model"] = ["4", 0]
        prompt = {"prompt": "photo", "negative": "bad", "seed": 1,
                  "episode": "TEST01", "scene": "S01"}
        with self.assertRaisesRegex(ValueError, "does not use"):
            comfy_workflow(prompt, template, "sdxl.safetensors", 640, 1152,
                           reference_image="approved.png")


if __name__ == "__main__":
    unittest.main()
