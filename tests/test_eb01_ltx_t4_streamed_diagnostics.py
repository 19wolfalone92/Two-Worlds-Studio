"""CPU-only regression for the private LTX T4 stage and logging safety."""
import ast
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"tools"/"kaggle_eb01_ltx2b_s02_fallback.py"

class TestLtxT4StageContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code=SOURCE.read_text(encoding="utf-8")

    def test_script_compiles_without_gpu(self):
        ast.parse(self.code)

    def test_encoder_before_transformer(self):
        s=self.code
        self.assertIn("self.vae = self.vae.to(device)",s)
        self.assertIn("self.vae = self.vae.cpu()",s)
        self.assertIn("self.transformer = self.transformer.to(device)",s)
        self.assertLess(s.index("self.vae = self.vae.cpu()"),s.index("self.transformer = self.transformer.to(device)"))
        self.assertIn("self.vae = self.vae.to(device)\\n            image = vae_decode",s)

    def test_no_hidden_enhancer(self):
        self.assertIn('threshold,"prompt_enhancement_words_threshold: 0"',self.code)
        self.assertIn("distilled_native_steps",self.code)

    def test_live_diagnostics_and_bounded_gpu(self):
        # Superseded earlier selector loop which leaked raw model log chunks
        # to the public Github console; the new helper is strictly private.
        self.assertIn("run_native_bounded(",self.code)
        self.assertIn("deadline_seconds=1400",self.code)
        self.assertIn("EB01_STAGE_",self.code)
        self.assertNotIn("print(chunk.decode",self.code)
        helper=(ROOT/"tools"/"eb01_bounded_inference.py").read_text()
        self.assertIn("subprocess.Popen(",helper)
        self.assertIn("start_new_session=True",helper)
        self.assertIn("os.killpg",helper)
        self.assertIn("latest_stage(",helper)
        self.assertIn('log_path.open("wb")',helper)
        self.assertIn("InferenceTimeBudgetExpired",helper)

    def test_no_auto_retry_or_publish(self):
        self.assertIn('cfg.get("new_gpu_jobs")==1',self.code)
        self.assertIn('cfg.get("allow_social_publish") is False',self.code)
        self.assertIn('"whole_movie_ready":False',self.code)
        self.assertNotIn("kaggle kernels push",self.code)

if __name__=="__main__":
    unittest.main()
