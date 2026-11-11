"""Regression tests for executable local ComfyUI model integration (no GPU/network)."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

from PIL import Image

from studio.comfy_neural import (
    MODEL_FILES, ComfyLocal, NeuralBlocked, _server, generate, import_take,
    template_workflow, wan22_workflow,
)
from studio.hybrid_master import check_manifest
from studio.release_gate import sha256


class GraphAndNetworkTests(unittest.TestCase):
    def test_official_wan_graph_has_image_branch_and_native_savevideo(self):
        graph = wan22_workflow(
            "source.png",
            "A character speaks naturally, turns her entire head and moves her shoulders.",
            42, width=576, height=1024, frames=121, steps=50,
        )
        self.assertEqual(graph["55"]["inputs"]["start_image"], ["56", 0])
        self.assertEqual(graph["56"]["inputs"]["image"], "source.png")
        self.assertEqual(graph["37"]["inputs"]["unet_name"], MODEL_FILES["wan22-5b"][0])
        self.assertEqual(graph["3"]["inputs"]["steps"], 50)
        self.assertEqual(graph["3"]["inputs"]["seed"], 42)
        self.assertEqual(graph["57"]["inputs"]["fps"], 24)
        self.assertEqual(graph["58"]["class_type"], "SaveVideo")
        self.assertTrue(all(isinstance(k, str) and isinstance(v, dict) and
                            "class_type" in v and isinstance(v["inputs"], dict)
                            for k, v in graph.items()))

    def test_missing_arguments_and_slideshow_workaround_blocked(self):
        prompt = "A figure moves smoothly, looks over their shoulder in a cinematic scene."
        for bad in [(544, 1024, 121), (576, 1024, 122), (576, 1024, 1)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                wan22_workflow("ref.png", prompt, 42, width=bad[0], height=bad[1], frames=bad[2])

    def test_loopback_only_even_with_redirect_or_http_credentials(self):
        for origin in (
            "http://example.com:8188", "http://169.254.169.254:8188",
            "https://127.0.0.1:8188", "http://x:y@127.0.0.1:8188",
            "http://127.0.0.1:8188/evil", "http://127.0.0.1",
        ):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                _server(origin)
        self.assertEqual(_server("http://127.0.0.1:8188"), "http://127.0.0.1:8188")
        self.assertEqual(_server("http://localhost:8188/"), "http://localhost:8188")

    def test_template_rejects_ui_format_and_missing_bindings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            graph = root / "graph.json"
            bindings = root / "bindings.json"
            graph.write_text(json.dumps({"nodes": [{"type": "SaveVideo"}]}))
            bindings.write_text("{}")
            with self.assertRaises(ValueError):
                template_workflow(graph, bindings, image_name="ref.png", prompt="A true test",
                                  seed=42, width=576, height=1024, frames=17)
            graph.write_text(json.dumps({
                "1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
                "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "A test"}},
                "3": {"class_type": "SaveVideo", "inputs": {"video": ["2", 0]}}
            }))
            with self.assertRaisesRegex(ValueError, "Missing valid API workflow binding"):
                template_workflow(graph, bindings, image_name="ref.png", prompt="A real action",
                                  seed=42, width=576, height=1024, frames=17)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class NeuralClipRoundTripTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.reference = self.root / "original.png"
        Image.new("RGB", (576, 1024), (30, 40, 55)).save(self.reference)
        self.source = self.root / "generated_temporary.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi",
            "-i", "color=c=blue:s=576x1024:r=24", "-frames:v", "17",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(self.source)
        ], check=True)
        self.output = self.root / "neural.mp4"
        self.prompt = "A woman naturally turns her entire body, with realistic shoulders and clothing."

    class FakeComfy:
        def __init__(self, source):
            self.source = source
            self.uploads = 0
            self.submissions = []

        def upload_image(self, source):
            self.uploads += 1
            return "tw_test_reference.png"

        def submit(self, workflow):
            self.submissions.append(workflow)
            return str(uuid.uuid4())

        def result(self, prompt_id, **kwargs):
            return {"outputs": {"58": {"videos": [{
                "filename": "test.mp4", "type": "output", "subfolder": ""
            }]}}}

        def download_video(self, job, destination):
            return {"temporary": self.source, "origin_filename": "test.mp4"}

    def _generate(self):
        client = self.FakeComfy(self.source)
        outcome = generate(
            "wan22-5b", reference=self.reference, output=self.output,
            prompt=self.prompt, seed=5, host="http://127.0.0.1:8188",
            width=576, height=1024, frames=17, steps=30, timeout=120,
            execute=True, accepted_license=True, client=client,
        )
        self.assertEqual(client.uploads, 1)
        self.assertEqual(client.submissions[0]["55"]["inputs"]["length"], 17)
        self.assertEqual(client.submissions[0]["56"]["inputs"]["image"], "tw_test_reference.png")
        return outcome

    def test_actual_comfy_receipt_imported_into_studio_master_with_source_lock(self):
        outcome = self._generate()
        self.assertEqual(outcome["status"], "GENERATED_REVIEW_REQUIRED")
        self.assertFalse(outcome["human_approved"])
        self.assertTrue(self.output.exists())
        receipt = self.output.with_suffix(".receipt.json")
        self.assertTrue(receipt.exists())
        accepted = import_take(self.output, receipt, self.reference, duration=17 / 24)
        self.assertEqual(accepted["model"], "wan22-5b")
        data = {
            "schema": 1, "delivery_profile": "mobile",
            "visual_profile": "photoreal_cinematic",
            "shots": [{
                "clip": self.output.name, "duration": 17 / 24,
                "clip_take": {
                    "engine": "neural_comfy", "visual_profile": "photoreal_cinematic",
                    "sha256": sha256(self.output),
                    "receipt": receipt.name, "reference": self.reference.name,
                },
            }],
        }
        plan = check_manifest(data, self.root)
        self.assertEqual(plan["shots"][0]["animation_engine"], "neural_comfy")
        self.assertEqual(plan["shots"][0]["motion_receipt"]["model"], "wan22-5b")
        data.update({"editorial_approved": True, "rights_approved": True})
        data["shots"][0]["approved"] = True
        with self.assertRaisesRegex(ValueError, "independent character-motion QC"):
            check_manifest(data, self.root, master=True)
        Image.new("RGB", (576, 1024), (255, 0, 0)).save(self.reference)
        with self.assertRaisesRegex(ValueError, "locked reference"):
            import_take(self.output, receipt, self.reference, duration=17 / 24)

    def test_plan_only_never_contacts_gpu(self):
        result = generate(
            "wan22-5b", reference=self.reference, output=self.output, prompt=self.prompt,
            seed=42, host="http://127.0.0.1:8188", width=576, height=1024,
            frames=17, steps=30, timeout=120, execute=False, accepted_license=True,
        )
        self.assertEqual(result["status"], "PLANNED_NO_GPU")
        self.assertFalse(self.output.exists())
        self.assertTrue(self.source.exists())

    def test_explicit_model_license_required(self):
        with self.assertRaises(NeuralBlocked):
            generate("wan22-5b", reference=self.reference, output=self.output,
                     prompt=self.prompt, seed=3, host="http://127.0.0.1:8188",
                     width=576, height=1024, frames=17, steps=30, timeout=120,
                     execute=False, accepted_license=False)


if __name__ == "__main__":
    unittest.main()
