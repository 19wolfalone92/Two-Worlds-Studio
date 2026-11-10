"""Verify temporal QC on decoded MP4s and protect authored optics boundaries."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from studio.cinematic import CinematicPanel
from studio.cinematic_lighting import integrate_exposure, validate_lens
from studio.facial_actor import MouthMesh
from studio.hybrid_master import produce
from studio.smooth_camera import encode_rgb_frames
from studio.video_qc import inspect_temporal, validate_regions


class OpticsTests(unittest.TestCase):
    def test_distance_and_aperture_affect_defocus(self):
        layers = [{"distance_m": 1.8}, {"distance_m": 6}]
        profile = {"focal_mm": 35, "focus_m": 1.8, "aperture": 4}
        lens = validate_lens(profile, layers)
        long = validate_lens({**profile, "focal_mm": 85}, layers)
        stopped_down = validate_lens({**profile, "focal_mm": 85, "aperture": 8}, layers)
        self.assertEqual(lens.blur_radius(1.8, 1080), 0)
        self.assertGreater(long.blur_radius(6, 1080), lens.blur_radius(6, 1080))
        self.assertGreater(long.blur_radius(6, 1080), stopped_down.blur_radius(6, 1080))
        with self.assertRaisesRegex(ValueError, "distance_m"):
            validate_lens(profile, [{}])
        with self.assertRaisesRegex(ValueError, "shutter_angle"):
            validate_lens({**profile, "shutter_angle": 360}, layers)

    def test_exposure_integrates_light_and_respects_protected_windows(self):
        black = Image.new("RGB", (1, 1))
        white = Image.new("RGB", (1, 1), "white")
        self.assertEqual(integrate_exposure(black, black, white).getpixel((0, 0)), (113,) * 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (288, 512), "darkgrey").save(root / "room.png")
            config = {"layers": [{"role": "background", "path": "room.png", "distance_m": 6}],
                      "optics": {"focal_mm": 85, "focus_m": 6, "aperture": 4, "shutter_angle": 180}}
            region = {"id": "clue", "layer": 0, "box": [.4, .4, .6, .6], "start": .25, "end": .75}
            scene = {"cinematic": config, "duration": 1., "protected_regions": [region]}
            panel = CinematicPanel(scene, root, 288, 512)
            with patch.object(panel, "frame", wraps=panel.frame) as frame:
                panel.exposed_frame(15, 30, 30)
                self.assertEqual(frame.call_count, 1)
                frame.reset_mock()
                panel.exposed_frame(6, 30, 30)
                self.assertEqual(frame.call_count, 3)
                self.assertAlmostEqual(frame.call_args_list[-1].kwargs["seconds"], .2 + 1 / 120)
            config["optics"]["focus_m"] = 1
            with self.assertRaisesRegex(ValueError, "defocus a protected region"):
                CinematicPanel(scene, root, 288, 512)

    def test_photographic_lip_shape_keeps_rest_pose_and_alpha(self):
        head = Image.new("RGBA", (200, 200), (140, 93, 72, 255))
        rig = SimpleNamespace(parts={"head": ({}, (head, (0, 0, 200, 200)))}, size=head.size)
        mesh = MouthMesh(rig, {"seam": [[.25, .60], [.50, .59], [.75, .60]], "opening_peak": .36})
        self.assertIs(mesh.frame((0., 0., 0.)), head)
        posed = np.asarray(mesh.frame((1., 0., 0.)))
        self.assertTrue((posed[:, :, 3] == 255).all())
        self.assertEqual(tuple(posed[70, 86, :3]), (140, 93, 72))
        self.assertLess(posed[126, 86, 0], posed[126, 134, 0])
        with self.assertRaisesRegex(ValueError, "opening_peak"):
            MouthMesh(rig, {"seam": [[.25, .60], [.50, .59], [.75, .60]], "opening_peak": .1})

    def test_required_source_detail_is_checked_before_compositing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (288, 512), "darkgrey").save(root / "room.png")
            scene = {"duration": 1., "cinematic": {"layers": [
                {"role": "background", "path": "room.png", "required_focus_box": [.4, .4, .6, .6]},
            ]}}
            with self.assertRaisesRegex(ValueError, "required art detail is soft or transparent"):
                CinematicPanel(scene, root, 288, 512)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class TemporalQCProductionTests(unittest.TestCase):
    size = (288, 512)
    frames = 30
    fps = 30
    region = {"id": "test-face", "kind": "face", "box": [.12, .18, .73, .52], "motion_required": True}

    def encode(self, path, *, held=False, pulses=()):
        width, height = self.size
        rng = np.random.default_rng(123)
        texture = rng.integers(55, 190, (height // 4, width // 4), dtype=np.uint8)
        image = Image.fromarray(texture).resize(self.size, Image.Resampling.BICUBIC)

        def images():
            for index in range(self.frames):
                offset = min(index, 7) if held else index
                shifted = np.roll(np.asarray(image, dtype=np.int16), offset, axis=1)
                if index in pulses:
                    shifted += 22
                yield Image.fromarray(shifted.clip(0, 255).astype(np.uint8)).convert("RGB")

        encode_rgb_frames(path, *self.size, self.fps, self.frames, images())

    def test_compensates_motion_and_reports_an_exposure_pulse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.mp4"
            self.encode(path)
            report = inspect_temporal(path, [self.region], *self.size, self.fps, self.frames)
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["anatomy_requires_review"])
            self.assertEqual(report["regions"][0]["frames_checked"], 30)
            path = path.with_name("pulse.mp4")
            self.encode(path, pulses=(12,))
            report = inspect_temporal(path, [self.region], *self.size, self.fps, self.frames)
            self.assertEqual(report["status"], "REVIEW REQUIRED")
            self.assertIn("exposure_oscillation", {e["code"] for e in report["regions"][0]["events"]})
            with self.assertRaisesRegex(ValueError, "decoded"):
                inspect_temporal(path, [self.region], *self.size, self.fps, 29)
            path = path.with_name("strobe.mp4")
            self.encode(path, pulses=(12, 14, 16))
            report = inspect_temporal(path, [self.region], *self.size, self.fps, self.frames)
            self.assertEqual(report["status"], "BLOCKED")
            allowed = {**self.region, "allow_exposure_change": True}
            report = inspect_temporal(path, [allowed], *self.size, self.fps, self.frames)
            self.assertEqual(report["status"], "PASS")

    def test_frozen_source_blocks_ordinary_production_and_keeps_output_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.encode(root / "shot.mp4", held=True)
            shot = {"clip": "shot.mp4", "duration": 1., "qc_regions": [self.region]}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema": 1, "width": 288, "height": 512,
                                            "fps": 30, "shots": [shot]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "BLOCKED: shot 1, test-face, unexpected_hold"):
                produce(manifest, root / "out.mp4")
            self.assertEqual({p.name for p in root.iterdir()}, {"shot.mp4", "manifest.json"})
            with self.assertRaisesRegex(ValueError, "geometry or policy"):
                validate_regions([self.region, self.region], self.size)


if __name__ == "__main__":
    unittest.main()
