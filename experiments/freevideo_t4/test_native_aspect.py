import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
from experiments.freevideo_t4.native_aspect_bridge import wan22_workflow, _reference
from experiments.freevideo_t4.adaptive_pipeline import native_3x4_geometry
from experiments.freevideo_t4.hybrid_engine import prepare

class NativeAspectTests(unittest.TestCase):
    def test_geometry_and_native_graph(self):
        for stage in ("preview","candidate","final"):
            for budget in (8,13):
                w,h=native_3x4_geometry(stage,budget)
                self.assertEqual(w*4,h*3)
                self.assertEqual((w%32,h%32),(0,0))
                graph=wan22_workflow("test.png","A seated man gently moves his head.",42,width=w,height=h,frames=121,steps=24)
                self.assertEqual(graph["55"]["inputs"]["width"],w)
                self.assertEqual(graph["55"]["inputs"]["height"],h)
        with self.assertRaises(ValueError):
            wan22_workflow("test.png","A seated man gently moves his head.",42,width=432,height=768)
    def test_real_cli_dry_run_all_profiles(self):
        import json, subprocess, sys
        from experiments.freevideo_t4.adaptive_pipeline import choose_plan, simulate_t4
        with TemporaryDirectory() as tmp:
            source=Path(tmp)/"source.png"
            Image.new("RGB",(600,800),(123,45,67)).save(source)
            plan=choose_plan(simulate_t4())
            for stage in ("preview","candidate"):
                for budget in (8,13):
                    w,h=native_3x4_geometry(stage,budget)
                    cmd=[sys.executable,"-m","experiments.freevideo_t4.native_aspect_bridge","--model","wan22-5b",
                         "--reference",str(source),"--output",str(Path(tmp)/"take.mp4"),"--prompt","A seated man gently moves his head.",
                         "--seed","42","--width",str(w),"--height",str(h),"--frames",str(plan[stage]["frames"]),
                         "--steps",str(plan[stage]["steps"]),"--accept-model-license"]
                    p=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
                    self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                    self.assertEqual(json.loads(p.stdout)["status"],"PLANNED_NO_GPU")
    def test_native_decode_contract_real_mp4(self):
        import shutil, subprocess
        from experiments.freevideo_t4.native_aspect_bridge import _probe, NeuralBlocked
        self.assertTrue(shutil.which("ffmpeg"))
        self.assertTrue(shutil.which("ffprobe"))
        with TemporaryDirectory() as tmp:
            for w,h in ((384,512),(480,640),(432,576),(432,768)):
                path=Path(tmp)/f"fixture-{w}-{h}.mp4"
                subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-f","lavfi","-i",
                                f"color=c=gray:s={w}x{h}:r=24","-frames:v","25","-c:v","libx264",
                                "-pix_fmt","yuv420p","-threads","1",str(path)],check=True,timeout=30)
                if w*4==h*3:
                    info=_probe(path)
                    self.assertEqual((info["width"],info["height"],info["frames"],info["fps"]),(w,h,25,"24"))
                else:
                    with self.assertRaises(NeuralBlocked):
                        _probe(path)
    def test_no_padding_and_full_original(self):
        import json
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"source.png"
            Image.new("RGB",(600,800),(123,45,67)).save(source)
            result=prepare(source,root/"job","native-aspect-test","A seated man gently moves his head.",size=(1152,1536))
            job=json.loads(Path(result["job_file"]).read_text())
            self.assertEqual(job["framing"],"native_3x4")
            self.assertEqual(job["content_origin"],[0,0])
            self.assertEqual(job["resized_content_size"],[1152,1536])
            self.assertEqual(_reference(Path(job["reference_path"]))["orientation"],"3:4")
            with Image.open(job["reference_path"]) as img:
                self.assertEqual(img.getpixel((0,0)),(123,45,67))

if __name__=="__main__":
    unittest.main()
