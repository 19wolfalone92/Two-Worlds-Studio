import json
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from studio.blender_clip import _load_contacts, render_clip
from studio.cli import run_engine
from studio.release_gate import sha256
from studio.shorts_factory import probe


class BlenderClipTests(unittest.TestCase):
    def test_contact_windows_must_cover_rendered_frames_and_use_meter_tolerances(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "contact.json"
            row = {"name": "held", "marker": "Finger", "target": "Handle",
                   "start_frame": 2, "end_frame": 4, "max_gap_m": .004, "max_slip_m": .001}
            manifest.write_text(json.dumps({"schema": 1, "contacts": [row]}))
            self.assertEqual(_load_contacts(manifest, 2, 4)[0], [row])
            for field, value in (("end_frame", 5), ("max_gap_m", float("nan")), ("marker", "Handle")):
                manifest.write_text(json.dumps({"schema": 1, "contacts": [{**row, field: value}]}))
                with self.assertRaisesRegex(ValueError, "Contact windows"):
                    _load_contacts(manifest, 2, 4)

    @unittest.skipUnless(all(shutil.which(x) for x in ("blender", "ffmpeg", "ffprobe")),
                         "Blender 4.3+ and FFmpeg required")
    def test_actual_blender_contact_detects_detachment_and_slip_in_meter_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, manifest = root / "scene.blend", root / "contact.json"
            setup = "\n".join((
                "import bpy",
                "scene = bpy.context.scene",
                "scene.cycles.use_denoising = False",
                "scene.unit_settings.scale_length = .01",
                "for name, positions in {'Handle': [(2, 0.), (4, .2)],"
                " 'Finger': [(2, .2), (4, .4)], 'Loose': [(2, .2), (4, .8)],"
                " 'Slip': [(2, .2), (4, 0.)]}.items():",
                "    marker = bpy.data.objects.new(name, None)",
                "    scene.collection.objects.link(marker)",
                "    for frame, x in positions:",
                "        marker.location.x = x",
                "        marker.keyframe_insert('location', frame=frame)",
                f"bpy.ops.wm.save_as_mainfile(filepath={str(source)!r})",
            ))
            subprocess.run(["blender", "--background", "--factory-startup", "--python-exit-code", "10",
                            "--python-expr", setup], capture_output=True, check=True, timeout=60)
            contact = {"name": "grip", "marker": "Finger", "target": "Handle",
                       "start_frame": 2, "end_frame": 4, "max_gap_m": .003, "max_slip_m": .001}
            manifest.write_text(json.dumps({"schema": 1, "contacts": [contact]}))
            result = render_clip(source, root / "held.mp4", first_frame=2, frames=3, width=180,
                                 samples=4, threads=2, contact_manifest=manifest)
            report, = result["contact_qc"]["windows"]
            self.assertEqual(report["frames_checked"], 3)
            self.assertAlmostEqual(report["peak_gap_m"], .002, places=5)
            self.assertLess(report["peak_slip_m"], 1e-7)
            for marker in ("Loose", "Slip"):
                manifest.write_text(json.dumps({"schema": 1, "contacts": [{**contact, "marker": marker}]}))
                rejected = root / f"{marker}.mp4"
                with self.assertRaisesRegex(ValueError, "Contact failed at frame"):
                    render_clip(source, rejected, first_frame=2, frames=3, width=180,
                                samples=4, threads=2, contact_manifest=manifest)
                self.assertFalse(rejected.exists())

    def test_offline_plan_and_render_bounds(self):
        plan = run_engine("blender", ["--frames", "66"])
        self.assertEqual(plan["module"], "studio.blender_clip")
        self.assertFalse(plan["actual_execution"])
        with self.assertRaisesRegex(ValueError, "render bounds"):
            render_clip(Path("scene.blend"), Path("take.mp4"), frames=301)
        with self.assertRaisesRegex(ValueError, "9:16"):
            render_clip(Path("scene.blend"), Path("take.mp4"), frames=3, width=181)
        with self.assertRaisesRegex(ValueError, "distinct collision mesh names"):
            render_clip(Path("scene.blend"), Path("take.mp4"), frames=3,
                        collision_pairs=[("Skin", "Skin")])

    def test_external_asset_manifest_rejects_missing_and_out_of_project_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, manifest, output = (root / name for name in ("scene.blend", "assets.json", "shot.mp4"))
            source.touch()
            for name, message in (("../private.png", "relative paths"),
                                  ("expression_0001.png", "missing or outside")):
                manifest.write_text(json.dumps({"schema": 1, "assets": {name: "0" * 64}}))
                with self.assertRaisesRegex(ValueError, message):
                    render_clip(source, output, frames=1, asset_manifest=manifest)
            self.assertFalse(output.exists())

    def test_external_asset_changed_during_render_blocks_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, manifest, output = (root / name for name in ("scene.blend", "assets.json", "shot.mp4"))
            source.touch()
            asset = root / "expression_0001.png"
            Image.new("RGB", (4, 4)).save(asset)
            manifest.write_text(json.dumps({"schema": 1, "assets": {asset.name: sha256(asset)}}))

            def render(args, **kwargs):
                temporary = Path(args[args.index("--python") + 1]).parent
                Image.new("RGB", (180, 320)).save(temporary / "shot_0001.png")
                Image.new("RGB", (4, 4), "white").save(asset)
                return subprocess.CompletedProcess(args, 0)

            with (patch("studio.blender_clip.shutil.which", return_value="/bin/true"),
                  patch("studio.blender_clip.subprocess.run", side_effect=render),
                  patch("studio.blender_clip.encode_rgb_frames",
                        side_effect=lambda path, *args: path.write_bytes(b"draft"))):
                with self.assertRaisesRegex(ValueError, "different SHA-256"):
                    render_clip(source, output, frames=1, width=180, asset_manifest=manifest)
            self.assertFalse(output.exists())

    @unittest.skipUnless(all(shutil.which(x) for x in ("blender", "ffmpeg", "ffprobe")),
                         "Blender 4.3+ and FFmpeg required")
    def test_real_blender_frames_are_encoded_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "scene.blend", root / "take.mp4"
            passes = root / "passes.zip"
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
                "bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, -1.2))",
                "floor = bpy.context.object",
                "floor.name = 'Floor'",
                "floor.is_shadow_catcher = True",
                "bpy.ops.mesh.primitive_cube_add(size=1, location=(1., .2, .2))",
                "probe = bpy.context.object",
                "probe.name = 'Crossing'",
                "probe.hide_render = True",
                "mesh = bpy.data.meshes.new('Internal crossing test')",
                "points = [tuple(v.co) for v in cube.data.vertices]",
                "faces = [tuple(p.vertices) for p in cube.data.polygons]",
                "mesh.from_pydata(points + [(x+1.2,y+.45,z+.2) for x,y,z in points], [], "
                "faces + [tuple(i+len(points) for i in p) for p in faces])",
                "overlap = bpy.data.objects.new('Internal crossing', mesh)",
                "scene.collection.objects.link(overlap)",
                "overlap.hide_render = True",
                f"bpy.ops.wm.save_as_mainfile(filepath={str(source)!r})",
            ))
            subprocess.run(["blender", "--background", "--factory-startup",
                            "--python-exit-code", "10", "--python-expr", setup],
                           capture_output=True, timeout=60, check=True)
            result = render_clip(source, output, first_frame=2, frames=3, width=180,
                                 samples=4, threads=2, passes_output=passes,
                                 collision_pairs=[("Cube", "Floor")], self_collision_meshes=["Cube"])
            self.assertEqual(result["geometry_qc"],
                             {"frames_checked": 3, "mesh_pairs": 1, "self_collision_meshes": 1})
            self.assertEqual(result["passes"]["sha256"], sha256(passes))
            self.assertIn("ShadowCatcher", result["passes"]["channels"])
            with zipfile.ZipFile(passes) as bundle:
                self.assertEqual(bundle.namelist(),
                                 ["layers_0002.exr", "layers_0003.exr", "layers_0004.exr", "manifest.json"])
                self.assertIsNone(bundle.testzip())
                header = bundle.read("layers_0002.exr")[:8192]
                for name in (b"Beauty.", b"Depth.", b"Normal.", b"ShadowCatcher.", b"GlossyIndirect."):
                    self.assertIn(name, header)
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
            blocked, blocked_passes = root / "blocked.mp4", root / "blocked.zip"
            with self.assertRaisesRegex(ValueError, "Mesh collision at frame 2"):
                render_clip(source, blocked, first_frame=2, frames=3, width=180,
                            samples=4, threads=2, passes_output=blocked_passes,
                            collision_pairs=[("Cube", "Crossing")])
            self.assertFalse(blocked.exists())
            self.assertFalse(blocked_passes.exists())
            with self.assertRaisesRegex(ValueError, "frame 2: Internal crossing"):
                render_clip(source, blocked, first_frame=2, frames=3, width=180, samples=4,
                            self_collision_meshes=["Internal crossing"])
            self.assertFalse(blocked.exists())


if __name__ == "__main__":
    unittest.main()
