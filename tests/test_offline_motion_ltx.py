import tempfile
import unittest
from pathlib import Path
from PIL import Image
from studio.offline_motion_ltx import prepare, execute, MotionBlocked, yaml_override

PROMPT=("An articulated friendly skeleton turns his head and speaks to a brunette "
        "artist in a candlelit studio. Slow moving camera, expressive jaw and eyes.")

class LTXMotionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        Image.new("RGB",(512,896),(32,24,30)).save(self.base/"ref.png")

    def test_missing_weights_block_without_download(self):
        planned=prepare(self.base/"official",self.base/"weights",self.base/"ref.png",
                        self.base/"output",PROMPT)
        self.assertIn("checkpoint",planned["missing"])
        with self.assertRaises(MotionBlocked):
            execute(planned,run_gpu=False)

    def test_wrong_frame_count_fails(self):
        with self.assertRaises(MotionBlocked):
            prepare(self.base,self.base,self.base/"ref.png",
                    self.base/"output",PROMPT,frames=80)

    def test_private_output_not_in_public_repo(self):
        with self.assertRaises(MotionBlocked):
            prepare(self.base,self.base,self.base/"ref.png",
                    self.base/"output",PROMPT)

    def test_reviewed_config_only(self):
        text=yaml_override("checkpoint_path: OLD\nprecision: bfloat16\n",
                           "checkpoint_path",self.base/"local.safetensors")
        self.assertIn("local.safetensors",text)
        with self.assertRaises(MotionBlocked):
            yaml_override("precision: bfloat16\n","checkpoint_path",self.base/"x")

if __name__=="__main__":
    unittest.main()
