import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from studio.cinematic import CinematicPanel
from studio.hybrid_master import check_manifest, render_segment, shot_key
from studio.kokoro_tts import render_jobs
from studio.release_gate import sha256
from studio.smooth_camera import encode_rgb_frames
from studio.video_qc import inspect_temporal


class RegisteredPerformanceTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.size = (288, 512)
        Image.new("RGB", self.size, (30, 45, 60)).save(self.root / "room.png")
        for part, bounds in (("head", (72, 115, 198, 260)), ("body", (55, 240, 228, 511))):
            image = Image.new("RGBA", self.size)
            ImageDraw.Draw(image).rectangle(bounds, fill=(160, 100, 64, 255))
            image.save(self.root / (part + ".png"))
        self.patch = self.root / "patch.png"
        Image.new("RGBA", (24, 16), (20, 208, 94, 255)).save(self.patch)
        self.take = self.root / "take.json"
        self.take.write_text(json.dumps({
            "schema": 1, "fps": 12, "canvas": list(self.size), "box": [118, 171, 142, 187],
            "portrait_sha256": sha256(self.root / "head.png"),
            "head_sha256": sha256(self.root / "head.png"), "speech": None,
            "frames": [None] * 6 + [{"path": self.patch.name, "sha256": sha256(self.patch)}] * 6,
        }))
        self.layer = {
            "role": "character", "path": "head.png", "sha256": sha256(self.root / "head.png"),
            "rig": {"schema": 1, "parts": [
                {"name": "torso", "path": "body.png", "pivot": [.5, .5]},
                {"name": "head", "path": "head.png", "pivot": [.5, .5], "parent": "torso"},
            ], "tracks": {"head": [{"at": 0}, {"at": 1, "x": .10}]}},
            "facial": {"performance_take": {"path": self.take.name, "sha256": sha256(self.take)}},
        }
        static = {"x": 0, "y": 0, "zoom": 1}
        self.shot = {
            "image": "room.png", "duration": 1, "camera": "static",
            "cinematic": {"vignette": 0., "camera": {"start": static, "end": static}, "layers": [
                {"role": "background", "path": "room.png"}, self.layer]},
        }
        self.doc = {"schema": 1, "width": 288, "height": 512, "fps": 12,
                    "visual_profile": "photoreal_cinematic", "shots": [self.shot]}

    def test_take_stays_on_the_head_and_enforces_exact_fps(self):
        panel = CinematicPanel(self.shot, self.root, *self.size, fps=12)
        self.assertEqual(panel.performance_modes, ["registered_facial_take"])
        self.assertLess(panel.exposed_frame(0, 12, 12).getpixel((130, 175))[1], 140)
        end = panel.exposed_frame(11, 12, 12)
        self.assertGreater(end.getpixel((157, 175))[1], 180)
        self.assertLess(end.getpixel((125, 175))[1], 140)
        with self.assertRaisesRegex(ValueError, "frame clock"):
            CinematicPanel(self.shot, self.root, *self.size, fps=24)

    def test_pinned_patch_participates_in_cache_and_tampering_fails(self):
        plan = check_manifest(self.doc, self.root)
        key = shot_key(plan["shots"][0], self.shot, plan)
        Image.new("RGBA", (24, 16), (255, 0, 0, 255)).save(self.patch)
        self.assertNotEqual(key, shot_key(plan["shots"][0], self.shot, plan))
        panel = CinematicPanel(self.shot, self.root, *self.size, fps=12)
        with self.assertRaisesRegex(ValueError, "frame SHA-256"):
            panel.exposed_frame(6, 12, 12)

    def test_photo_auto_prefers_locked_actor_and_protects_dialogue_clock(self):
        self.shot["i2v"] = {"clip": "missing.mp4", "receipt": "missing.json"}
        shot = check_manifest(self.doc, self.root)["shots"][0]
        self.assertEqual(shot["animation_engine"], "registered_character_performance")
        self.assertEqual(shot["fallback"], "i2v_excluded_to_preserve_locked_portrait")
        manifest = json.loads(self.take.read_text())
        manifest["speech"] = {"wav_sha256": "invented"}
        self.take.write_text(json.dumps(manifest))
        self.layer["facial"]["performance_take"]["sha256"] = sha256(self.take)
        with self.assertRaisesRegex(ValueError, "WAV, text and voice offset"):
            CinematicPanel(self.shot, self.root, *self.size, fps=12)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class DecodedShotTests(unittest.TestCase):
    def test_clip_profiles_and_motion_must_have_the_full_footage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            clip = root / "motion.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=288x512:r=12",
                "-frames:v", "6", "-c:v", "libx264", str(clip),
            ], check=True)
            take = {"engine": "blender_3d", "visual_profile": "stylized_cinematic", "sha256": sha256(clip)}
            shot = {"clip": clip.name, "clip_take": take, "duration": 1, "shot_type": "close"}
            manifest = {"schema": 1, "width": 288, "height": 512, "fps": 12, "shots": [shot],
                        "visual_profile": "photoreal_cinematic"}
            with self.assertRaisesRegex(ValueError, "visual profile"):
                check_manifest(manifest, root)
            take["visual_profile"] = manifest["visual_profile"]
            with self.assertRaisesRegex(ValueError, "shorter than"):
                check_manifest(manifest, root)
            shot["duration"] = .5
            selected = check_manifest(manifest, root)["shots"][0]
            receipt = render_segment(selected, root / "output.mp4", 288, 512, 12)
            self.assertEqual(selected["animation_engine"], "blender_3d")
            self.assertFalse(receipt["frame_padding_allowed"])

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "cinematic extra required")
    def test_optical_flow_distinguishes_global_zoom_from_actor_action(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            background = Image.new("RGB", (288, 512), (36, 41, 45))
            draw = ImageDraw.Draw(background)
            for y in range(20, 160, 17):
                for x in range(20, 255, 15):
                    draw.rectangle((x, y, x + 7, y + 8), fill=(110 + x // 5, 120, 160))
            actor = Image.new("RGB", (70, 100), (97, 74, 65))
            draw = ImageDraw.Draw(actor)
            for x in range(4, 65, 10):
                draw.ellipse((x, 9, x + 8, 26), fill=(210, 183, 90))
                draw.rectangle((x, 48, x + 7, 85), fill=(110, 204, 139))
            regions = [
                {"id": "camera", "kind": "camera", "box": [.04, .03, .92, .29]},
                {"id": "actor", "kind": "face", "box": [.26, .37, .74, .70],
                 "motion_windows": [{"start": 0, "end": 1}]},
            ]

            def make(i, action):
                frame = background.copy()
                frame.paste(actor, (93 + i * 2 if action else 93, 205))
                scale = 1 - .005 * i
                return frame.transform((288, 512), Image.Transform.AFFINE,
                                       (scale, 0, 144 * (1 - scale), 0, scale, 256 * (1 - scale)),
                                       Image.Resampling.BICUBIC)

            for action in (False, True):
                path = root / f"{action}.mp4"
                encode_rgb_frames(path, 288, 512, 12, 12, (make(i, action) for i in range(12)))
                report = inspect_temporal(path, regions, 288, 512, 12, 12)["optical_flow"]
                self.assertGreater(report["regions"][0]["camera_motion"]["pairs"], 8)
                codes = {event["code"] for event in report["regions"][1]["events"]}
                self.assertEqual("no_independent_action" in codes, not action)

    def test_kokoro_rejects_russian_even_in_a_dry_run(self):
        with self.assertRaisesRegex(ValueError, "English-only"):
            render_jobs([{"text": "Проба голоса."}], Path("."), dry_run=True)
