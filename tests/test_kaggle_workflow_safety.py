"""Static workflow contract: keep Kaggle API traffic safe and proof downloads private."""
from __future__ import annotations
import unittest
from studio.kaggle_api import ROOT

class KaggleWorkflowSafetyTests(unittest.TestCase):
    def test_shared_concurrency_prevents_parallel_budget_pressure(self):
        files=(".github/workflows/comfyui-kaggle.yml",
               ".github/workflows/comfyui-readback.yml",
               ".github/workflows/kaggle-api.yml",
               ".github/workflows/kaggle-connection-check.yml")
        for name in files:
            text=(ROOT/name).read_text(encoding="utf-8")
            with self.subTest(name=name):
                self.assertIn("group: two-worlds-kaggle-account",text)
                self.assertIn("cancel-in-progress: false",text)

    def test_all_production_readbacks_call_one_safe_filter(self):
        for name in (".github/workflows/comfyui-kaggle.yml",
                     ".github/workflows/comfyui-readback.yml",
                     ".github/workflows/kaggle-api.yml"):
            text=(ROOT/name).read_text(encoding="utf-8")
            with self.subTest(name=name):
                self.assertIn("studio.kaggle_api_limits output",text)
                self.assertNotIn("for delay in",text)
                self.assertNotIn("kaggle kernels output",text)
                self.assertNotIn("kaggle quota >/dev/null",text)

    def test_source_does_not_enable_auto_gpu_or_public_art(self):
        w=(ROOT/".github/workflows/comfyui-kaggle.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:",w)
        self.assertIn("ops/comfy_launch.trigger",w)
        self.assertIn("if: ${{ github.event.repository.private }}",w)
        self.assertNotIn("schedule:",w)
        read=(ROOT/".github/workflows/comfyui-readback.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:",read)
        self.assertNotIn("ops/comfy_verify.trigger",read)

if __name__=="__main__":
    unittest.main()
