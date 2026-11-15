"""Source-locked controlled LTX native-motion A/B. Tests do NOT simulate generation."""
from __future__ import annotations
import ast
import unittest
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "tools" / "kaggle_eb01_ltx2b_s02_fallback.py"


def native_run_config_script() -> str:
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    found = [node.value.value
             for node in ast.walk(tree)
             if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "script" for t in node.targets)
             and isinstance(node.value, ast.Constant)
             and isinstance(node.value.value, str)
             and "InferenceConfig(" in node.value.value]
    if len(found) != 1:
        raise AssertionError("Native Kaggle exact-call script not uniquely pinned")
    return found[0]


class RealLTXMotionExperimentTests(unittest.TestCase):
    def test_pinned_controlled_single_variable_motion_ab(self):
        script = native_run_config_script()
        ast.parse(script)
        # Only the official I2V image conditioning noise is experimentally
        # increased; all other source and model parameters stay unchanged.
        for signature in (
            "image_cond_noise_scale=0.15",
            "conditioning_strengths=[1.0]",
            "conditioning_start_frames=[0]",
            "seed=202611132",
            "num_frames=33",
            "frame_rate=24",
            "height=576,width=320",
            "offload_to_cpu=True",
            "prompt=os.environ['LTX_PROMPT']",
            "conditioning_media_paths=[os.environ['LTX_INPUT']]",
        ):
            self.assertIn(signature, script)
        self.assertNotIn("image_cond_noise_scale=0.025", script)
        self.assertNotIn("prompt_enhancement=True", script)

    def test_native_worker_produces_private_real_frame_motion_evidence(self):
        source = WORKER.read_text(encoding="utf-8")
        for must in (
            "from tools.eb01_true_gpu_clip_numbers import gray_motion",
            "numerical_motion=gray_motion(video)",
            '"motion_noise_conditioning":0.15',
            '"conditioning_strength_explicit":1.0',
            '"actor_motion_artistic_QA_proven":False',
            '"whole_movie_ready":False',
            "run_native_bounded(",
            "deadline_seconds=1400",
        ):
            self.assertIn(must, source)
        self.assertNotIn("kaggle kernels push", source)
        self.assertNotIn("print(chunk.decode", source)

    def test_no_claim_of_sufficient_lipsync_from_motion_proxy(self):
        source=WORKER.read_text(encoding="utf-8")
        self.assertIn('"audio_conditioned_lipsync_verified":False', source)
        self.assertIn('"human_artistic_approved":False', source)
        self.assertIn('"published":False', source)


if __name__ == "__main__":
    unittest.main()
