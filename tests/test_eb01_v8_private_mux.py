"""Private V8 compositor regressions. All video assets here are synthetic TEST fixtures.

These tests NEVER claim model inference, actor authenticity, or a finished film.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from studio.eb01_v8_private_mux import (
    PrivateMovieBlocked, safe_private_path, prepare_private_output_parent,
    secure_private_file, segment_filter, transcode_one, native33,
    final_frame_rgb_sha, ffprobe, assemble_private_v8, audio_mix,
    validate_voice_timing
)
from studio.eb01_v7_storyboard import VOICE_SECONDS
from studio.eb01_v8_40s_native_motion_plan import plan


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "real ffmpeg and ffprobe necessary")
class RealCodecFrameTests(unittest.TestCase):
    def test_exact_33_vs_32_original_frames_after_conditioning_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            source = base / "synthetic_33f_native_576x1024.mp4"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc2=size=576x1024:rate=24",
                 "-frames:v", "33", "-c:v", "libx264",
                 "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(source)],
                check=True, capture_output=True, timeout=90)
            native33(source)
            self.assertEqual(len(final_frame_rgb_sha(source)),64)
            for skipped, expected in ((False, 33), (True, 32)):
                out = base / f"synthetic_test_skip_{skipped}.mp4"
                transcode_one(source, out, skipped)
                v = [x for x in ffprobe(out)["streams"]
                     if x["codec_type"] == "video"][0]
                self.assertEqual(v["width"], 576)
                self.assertEqual(v["height"], 1024)
                self.assertEqual(v["avg_frame_rate"], "24/1")
                self.assertEqual(int(v["nb_read_frames"]), expected)

    def test_320x576_pilot_cannot_enter_native_v8_source_pool(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "synthetic_low_resolution_pilot.mp4"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc2=size=320x576:rate=24",
                 "-frames:v", "33", "-c:v", "libx264",
                 "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(source)],
                check=True, capture_output=True, timeout=60)
            with self.assertRaisesRegex(PrivateMovieBlocked, "320x576 upscale is not detail"):
                native33(source)

    def test_six_independent_voice_inputs_produce_full_length_audio_clock(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            wavs = {}
            for i, sid in enumerate(["S01", "S02", "S03", "S04", "S05", "S06"]):
                wave = base / f"{sid}.wav"
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y",
                     "-f", "lavfi", "-i", f"sine=frequency={300+i*65}:sample_rate=48000",
                     "-t", str(VOICE_SECONDS[sid]), "-ac", "1",
                     "-c:a", "pcm_s16le", str(wave)],
                    capture_output=True, check=True, timeout=30)
                wavs[sid] = wave
            merged = base / "SYNTHETIC_TEST_NOT_REAL_RUSSIAN_AUDIO.wav"
            timing = audio_mix(base, wavs, plan(), merged)
            self.assertEqual(timing["mix_sample_rate"], 48000)
            self.assertEqual(timing["samples_per_video_frame"], 2000)
            self.assertEqual(set(timing["voice_lines"]), set(wavs))
            for proof in timing["voice_lines"].values():
                self.assertEqual(proof["start_sample_48k"],
                                 proof["movie_start_frame"] * 2000)
                self.assertLessEqual(proof["duration_seconds"],
                                     proof["window_seconds"] + 1 / 24000)
            streams = [x for x in ffprobe(merged)["streams"]
                       if x["codec_type"] == "audio"]
            self.assertEqual(len(streams), 1)
            self.assertAlmostEqual(float(streams[0]["duration"]),
                                   plan()["final_duration_seconds"], delta=0.04)

    def test_overlong_voice_is_rejected_instead_of_spilling_into_next_scene(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            scene = next(s for s in plan()["scenes"] if s["scene_id"] == "S02")
            wave = base / "overlong_S02.wav"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y",
                 "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                 "-t", str(VOICE_SECONDS["S02"] + 0.25), "-ac", "1",
                 "-c:a", "pcm_s16le", str(wave)],
                capture_output=True, check=True, timeout=30)
            with self.assertRaisesRegex(PrivateMovieBlocked, "dialogue window"):
                validate_voice_timing("S02", wave, scene)

    def test_truncated_voice_is_rejected_instead_of_silently_passing(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            scene = next(s for s in plan()["scenes"] if s["scene_id"] == "S02")
            wave = base / "truncated_S02.wav"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y",
                 "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                 "-t", str(VOICE_SECONDS["S02"] - 0.25), "-ac", "1",
                 "-c:a", "pcm_s16le", str(wave)],
                capture_output=True, check=True, timeout=30)
            with self.assertRaisesRegex(PrivateMovieBlocked, "dialogue window"):
                validate_voice_timing("S02", wave, scene)

    def test_no_real_sources_never_creates_fake_final_movie(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "private"
            for folder in ("segments", "receipts", "voices"):
                (root / folder).mkdir(parents=True)
            outfile = Path(td) / "PRIVATE_NOT_CREATED.mp4"
            with self.assertRaisesRegex(PrivateMovieBlocked, "Missing genuine"):
                assemble_private_v8(root, outfile)
            self.assertFalse(outfile.exists())


class PrivateNoPublicReleaseTests(unittest.TestCase):
    def test_output_inside_public_github_workspace_denied(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            with patch.dict(os.environ, {"GITHUB_WORKSPACE": str(base)}):
                with self.assertRaises(PrivateMovieBlocked):
                    safe_private_path(base / "test_private.mp4")

    def test_local_git_checkout_denied_without_github_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            checkout = Path(td) / "public_checkout"
            (checkout / ".git").mkdir(parents=True)
            with patch.dict(os.environ, {"GITHUB_WORKSPACE": ""}):
                with self.assertRaisesRegex(PrivateMovieBlocked, "Git checkout"):
                    safe_private_path(checkout / "private" / "actor.mp4")

    def test_symlinked_parent_denied_even_when_target_is_private(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            vault = base / "vault"
            vault.mkdir()
            alias = base / "alias"
            try:
                alias.symlink_to(vault, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Directory symlinks unavailable")
            with self.assertRaisesRegex(PrivateMovieBlocked, "Symlinked"):
                safe_private_path(alias / "review.mp4")

    @unittest.skipUnless(os.name == "posix", "POSIX owner-only mode bits required")
    def test_new_review_directory_is_owner_only(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "owner-vault" / "review.mp4"
            parent = prepare_private_output_parent(output)
            self.assertEqual(parent.stat().st_mode & 0o777, 0o700)

    @unittest.skipUnless(os.name == "posix", "POSIX owner-only mode bits required")
    def test_shared_group_or_world_readable_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            vault = Path(td) / "shared-vault"
            vault.mkdir()
            vault.chmod(0o755)
            with self.assertRaisesRegex(PrivateMovieBlocked, "Group/world-readable"):
                prepare_private_output_parent(vault / "review.mp4")

    @unittest.skipUnless(os.name == "posix", "POSIX owner-only mode bits required")
    def test_final_review_file_is_owner_read_write_only(self):
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "review.mp4"
            movie.write_bytes(b"synthetic-not-real-media")
            movie.chmod(0o644)
            secure_private_file(movie)
            self.assertEqual(movie.stat().st_mode & 0o777, 0o600)

    def test_no_frame_duplication_or_generated_fps_in_segment_filter(self):
        for skip in (False, True):
            f = segment_filter(skip)
            self.assertIn("trim=start_frame=" + ("1" if skip else "0"), f)
            self.assertIn("setpts=PTS-STARTPTS", f)
            self.assertIn("force_original_aspect_ratio=increase", f)
            self.assertIn("crop=576:1024", f)
            self.assertNotIn("minterpolate", f)
            self.assertNotIn("fps=", f)
            self.assertNotIn("loop", f)

    def test_final_40_second_frame_math_is_only_a_plan(self):
        p = plan()
        self.assertEqual(p["new_true_inference_segment_count"], 30)
        self.assertEqual(p["final_unique_native_frames"], 967)
        self.assertEqual(p["original_russian_qwen_lines"], 6)
        self.assertEqual(p["required_native_width"], 576)
        self.assertEqual(p["required_native_height"], 1024)
        self.assertFalse(p["production_executed"])
        self.assertFalse(p["cinematic_owner_approved"])


if __name__ == "__main__":
    unittest.main()
