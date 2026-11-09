"""Offline regressions for the portrait film entry point and resumable rendering."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import shutil
import subprocess
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from studio import cli, production_queue
from studio.cinematic import CinematicPanel
from studio.hybrid_master import check_manifest, produce
from studio.lipsync import rhubarb_visemes, speech_activity
from studio.model_cache import cached_segment
from studio.release_gate import sha256

FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


class CinematicProductionTests(unittest.TestCase):
    size = (288, 512)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.background = self.art("background", (31, 62, 89, 255))
        self.body = self.art("body", None, (92, 222, 205, 398), (136, 92, 53, 255))
        head = self.art("head", None, (110, 140, 171, 235), (208, 147, 103, 255))
        arm = self.art("arm", None, (202, 242, 222, 337), (160, 55, 40, 255))
        self.rig = {"schema": 1, "parts": [
            {"name": "torso", "pivot": [.5, .6], **self.body},
            {"name": "head", "pivot": [.5, .43], "parent": "torso", **head},
            {"name": "arm", "pivot": [.73, .48], "parent": "torso", **arm},
        ], "tracks": {"head": [{"at": 0}, {"at": 1, "angle": 12}],
                      "arm": [{"at": 0}, {"at": 1, "angle": -23}]}}
        self.fixed_camera = {"start": {"zoom": 1}, "end": {"zoom": 1}}
        self.layer = {"role": "character", **self.body, "rig": self.rig}
        self.scene = {"duration": 1.2, "cinematic": {"camera": self.fixed_camera,
                      "layers": [{"role": "background", **self.background}, self.layer],
                      "vignette": 0}}
        self.shot = {"image": self.background["path"], "cinematic": self.scene["cinematic"],
                     "duration": 37 / 30, "camera": "static"}
        self.manifest = {"schema": 1, "width": 288, "height": 512, "fps": 30,
                         "language": "ru", "burn_subtitles": True, "shots": [self.shot]}

    def art(self, name, background, rect=None, fill=None):
        image = Image.new("RGBA", self.size, background or (0, 0, 0, 0))
        if rect:
            ImageDraw.Draw(image).rectangle(rect, fill=fill)
        target = self.root / f"{name}.png"
        image.save(target)
        return {"path": target.name, "sha256": sha256(target)}

    def wav(self, name="voice"):
        rate = 16000
        t = np.arange(16000) / rate
        pcm = np.where((t >= .18) & (t < .73), .32 * np.sin(t * 2 * math.pi * 289), 0)
        target = self.root / f"{name}.wav"
        with wave.open(str(target), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes((pcm * 32767).astype("<i2").tobytes())
        return target

    def attach_voice(self):
        voice = self.wav()
        open_plate = self.art("open", None, (130, 194, 150, 206), (215, 26, 50, 255))
        rest_plate = self.art("rest", None, (130, 194, 150, 206), (128, 73, 59, 255))
        track = {"schema": 1, "engine": "rhubarb_phonetic", "wav_sha256": sha256(voice),
                 "visemes": [{"at": .10, "shape": "D"}, {"at": .9, "shape": "rest"}]}
        (self.root / "mouth.json").write_text(json.dumps(track), encoding="utf-8")
        self.layer["facial"] = {"mouth": {"rest": rest_plate, "D": open_plate},
                                "viseme_track": "mouth.json"}
        self.scene["voice"] = {"path": "voice.wav", "offset_ms": 100}
        self.shot.update(voice="voice.wav", voice_offset=.1, dialogue_text="Я здесь.",
                         subtitle_cues=[{"start": .28, "end": .85, "text": "Я здесь."}])

    def test_rig_changes_actor_pixels_even_with_locked_camera(self):
        panel = CinematicPanel(self.scene, self.root, *self.size)
        self.assertEqual(panel.active_actors, 1)
        first, last = panel.frame(0), panel.frame(1)
        self.assertEqual(first.getpixel((10, 10)), last.getpixel((10, 10)))
        self.assertNotEqual(first.crop((90, 120, 266, 390)).tobytes(),
                            last.crop((90, 120, 266, 390)).tobytes())

    @unittest.skipUnless(FFMPEG, "FFmpeg and ffprobe required")
    def test_actual_wav_gates_visemes_during_pauses_and_honors_offset(self):
        self.attach_voice()
        panel = CinematicPanel(self.scene, self.root, *self.size)
        silent = panel.frame(0, seconds=.22)
        talking = panel.frame(0, seconds=.45)
        tail = panel.frame(0, seconds=.87)
        self.assertNotEqual(silent.getpixel((140, 200)), talking.getpixel((140, 200)))
        self.assertEqual(tail.getpixel((140, 200)), silent.getpixel((140, 200)))
        self.assertAlmostEqual(speech_activity(self.root / "voice.wav")["start"], .18, places=2)
        (self.root / "voice.wav").write_bytes(self.wav("other").read_bytes() + b"X")
        with self.assertRaisesRegex(ValueError, "exact WAV"):
            CinematicPanel(self.scene, self.root, *self.size)

    def test_protected_clue_rejects_camera_caption_and_layer_occlusion(self):
        region = {"id": "clue", "layer": 0, "box": [.3, .28, .45, .37], "start": 0., "end": 1.}
        scene = copy.deepcopy(self.scene)
        scene["cinematic"]["layers"] = scene["cinematic"]["layers"][:1]
        scene["protected_regions"] = [region]
        panel = CinematicPanel(scene, self.root, *self.size)
        panel.frame(.5, seconds=.6)
        scene["caption_windows"] = [(0., .9, .22, .43)]
        with self.assertRaisesRegex(ValueError, "captions"):
            CinematicPanel(scene, self.root, *self.size).frame(.5, seconds=.6)
        del scene["caption_windows"]
        cover = {"role": "foreground", **self.art("cover", (24, 33, 41, 255))}
        scene["cinematic"]["layers"].append(cover)
        with self.assertRaisesRegex(ValueError, "occludes"):
            CinematicPanel(scene, self.root, *self.size).frame(.5, seconds=.6)
        scene["cinematic"]["layers"].pop()
        region["box"] = [0, 0, .1, .2]
        with self.assertRaisesRegex(ValueError, "crops"):
            CinematicPanel(scene, self.root, *self.size).frame(.5, seconds=.6)

    @unittest.skipUnless(FFMPEG, "FFmpeg and ffprobe required")
    def test_i2v_import_checks_decoded_frames_hash_and_does_not_start_provider(self):
        take = self.root / "test_take.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-loop", "1", "-framerate", "30",
                        "-i", str(self.root / "background.png"), "-frames:v", "42",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(take)], check=True)
        from studio.neural_motion import CHECKPOINT, REPO

        proof = {"model": REPO, "checkpoint": CHECKPOINT,
                 "source_image_sha256": self.background["sha256"],
                 "sha256": sha256(take), "actual_neural_frames": 42,
                 "width": self.size[0], "height": self.size[1]}
        receipt = self.root / "take.json"
        receipt.write_text(json.dumps(proof), encoding="utf-8")
        self.shot["i2v"] = {"clip": take.name, "receipt": receipt.name}
        self.assertEqual(check_manifest(self.manifest, self.root)["shots"][0]["kind"], "i2v")
        proof["actual_neural_frames"] = 41
        receipt.write_text(json.dumps(proof), encoding="utf-8")
        fallback = check_manifest(self.manifest, self.root)["shots"][0]
        self.assertEqual((fallback["kind"], fallback["fallback"]),
                         ("layered", "i2v_take_missing_or_invalid"))
        self.shot["animation_backend"] = "i2v"
        with self.assertRaisesRegex(ValueError, "source-pinned"):
            check_manifest(self.manifest, self.root)

    @unittest.skipUnless(FFMPEG, "FFmpeg and ffprobe required")
    def test_cli_delivers_exact_frames_and_aac_from_wav_layers_and_timed_sound(self):
        self.attach_voice()
        self.manifest["shots"] += [{"image": "background.png", "duration": 19/30, "camera": "static"},
                                   {"image": "background.png", "duration": 17/30, "camera": "static"}]
        self.manifest["sound_stems"] = [{"bus": "sfx", "path": self.wav("hit").name,
                                         "gain": .16, "start": 1.16}]
        path = self.root / "film.json"
        path.write_text(json.dumps(self.manifest), encoding="utf-8")
        cache = self.root / "cache"
        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            self.assertEqual(cli.main(["produce", "--manifest", str(path), "--output",
                                       str(self.root / "film1.mp4"), "--cache", str(cache)]), 0)
        decoded = json.loads(report.getvalue())
        self.assertEqual(decoded["qc"]["technical_status"], "PASS")
        self.assertEqual(decoded["qc"]["status"], "REVIEW REQUIRED")
        self.assertEqual(decoded["qc"]["delivery"]["av_delivery"]["decoded_frames"], 73)
        self.assertEqual(decoded["shot_reports"][0]["lipsync_modes"], ["waveform_gated_visemes"])
        self.assertEqual(decoded["audio_mastering"]["buses"], {"dialogue": 1, "sfx": 1})
        replay = produce(path, self.root / "film2.mp4", cache=cache)
        self.assertTrue(all(shot["cached"] for shot in replay["shot_reports"]))
        self.assertIn("Я здесь.", (self.root / "film2.srt").read_text())
        self.assertFalse(decoded["published"])
        self.shot["dialogue_text"] = "Другие слова."
        with self.assertRaisesRegex(ValueError, "locked dialogue"):
            check_manifest(self.manifest, self.root)

    @unittest.skipUnless(FFMPEG, "FFmpeg and ffprobe required")
    def test_black_pixels_are_blocked_and_never_committed(self):
        black = self.art("black", (0, 0, 0, 255))
        self.manifest["shots"] = [{"image": black["path"], "duration": 1.2,
                                   "voice": self.wav().name, "caption": "Я здесь."}]
        self.manifest["sound_stems"] = []
        path = self.root / "broken.json"
        path.write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "BLOCKED.*(black|dark)"):
            produce(path, self.root / "broken-film.mp4")
        self.assertFalse((self.root / "broken-film.mp4").exists())


class CinematicResourceTests(unittest.TestCase):
    def test_rhubarb_rejects_overlap_and_keeps_silent_gaps(self):
        document = {"mouthCues": [{"start": .1, "end": .3, "value": "C"},
                                  {"start": .5, "end": .7, "value": "D"}]}
        self.assertEqual(rhubarb_visemes(document, .71)[1], {"at": .3, "shape": "rest"})
        document["mouthCues"][1]["start"] = .2
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            rhubarb_visemes(document, .71)

    def test_private_cache_repairs_corrupt_media_and_retains_interrupted_job(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            cache = folder / "cache"
            dest = folder / "shot.mp4"
            calls = []
            key = "1" * 64
            def render(path):
                calls.append(path)
                path.write_bytes(b"fixture payload")
                return {"rendered": True}
            self.assertFalse(cached_segment(cache, key, dest, render)[1])
            self.assertTrue(cached_segment(cache, key, dest, render)[1])
            (cache / f"{key}.mp4").write_bytes(b"corrupt")
            self.assertFalse(cached_segment(cache, key, dest, render)[1])
            self.assertEqual(len(calls), 2)
            interrupted = "2" * 64
            def fail(path):
                path.write_bytes(b"incomplete")
                raise InterruptedError("worker interrupted")
            with self.assertRaises(InterruptedError):
                cached_segment(cache, interrupted, dest, fail)
            self.assertFalse((cache / f"{interrupted}.mp4").exists())
            with contextlib.closing(production_queue.open_queue(cache / "jobs.sqlite3")) as db:
                lease = production_queue.claim(db, "new-worker", now=time.time()+65,
                                               job_id="segment."+interrupted)
                self.assertEqual(lease["attempt"], 2)


if __name__ == "__main__":
    unittest.main()
