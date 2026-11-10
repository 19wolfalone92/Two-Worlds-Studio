import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from studio.blender_clip import render_clip
from studio.cli import run_engine
from studio.release_gate import sha256
from studio.shorts_factory import probe


class BlenderClipTests(unittest.TestCase):
    def test_offline_plan_and_render_bounds(self):
        plan = run_engine("blender", ["--frames", "66"])
        self.assertEqual(plan["module"], "studio.blender_clip")
        self.assertFalse(plan["actual_execution"])
        with self.assertRaisesRegex(ValueError, "render bounds"):
            render_clip(Path("scene.blend"), Path("take.mp4"), frames=301)
        with self.assertRaisesRegex(ValueError, "9:16"):
            render_clip(Path("scene.blend"), Path("take.mp4"), frames=3, width=181)

    @unittest.skipUnless(all(shutil.which(x) for x in ("blender", "ffmpeg", "ffprobe")),
                         "Blender 4.3+ and FFmpeg required")
    def test_real_blender_frames_are_encoded_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "scene.blend", root / "take.mp4"
            setup = "\n".join((
                "import bpy",
                "scene = bpy.context.scene",
                "scene.render.engine = 'CYCLES'",
                "scene.cycles.use_denoising = False",
                "cube = bpy.data.objects['Cube']",
                "cube.rotation_euler.z = 0",
                "cube.keyframe_insert('rotation_euler', frame=2)",
                "cube.rotation_euler.z = .7",
                "cube.keyframe_insert('rotation_euler', frame=4)",
                f"bpy.ops.wm.save_as_mainfile(filepath={str(source)!r})",
            ))
            subprocess.run(["blender", "--background", "--factory-startup",
                            "--python-exit-code", "10", "--python-expr", setup],
                           capture_output=True, timeout=60, check=True)
            result = render_clip(source, output, first_frame=2, frames=3, width=180,
                                 samples=4, threads=2)
            streams = probe(output)["streams"]
            self.assertEqual(len(streams), 1)
            self.assertEqual(streams[0]["codec_name"], "h264")
            self.assertEqual((streams[0]["width"], streams[0]["height"]), (180, 320))
            self.assertEqual(streams[0]["avg_frame_rate"], "30/1")
            self.assertEqual(int(streams[0]["nb_frames"]), 3)
            self.assertEqual(result["clip_sha256"], sha256(output))
            decoded = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output),
                                               "-f", "framemd5", "-"], text=True, timeout=30)
            self.assertEqual(len({line.rsplit(",", 1)[-1] for line in decoded.splitlines()
                                  if not line.startswith("#")}), 3)
            with self.assertRaises(FileExistsError):
                render_clip(source, output, frames=3)
            self.assertEqual(result["clip_sha256"], sha256(output))


if __name__ == "__main__":
    unittest.main()
