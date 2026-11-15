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
    secure_private_file, segment_filter, transcode_one,
    ffprobe, assemble_private_v8, audio_mix
)
from studio.eb01_v8_40s_native_motion_plan import plan


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "real ffmpeg and ffprobe necessary")
class RealCodecFrameTests(unittest.TestCase):
    def test_exact_33_vs_32_original_frames_after_conditioning_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            source = base / "synthetic_33f.mp4"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc2=size=320x576:rate=24",
                 "-frames:v", "33", "-c:v", "libx264",
                 "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(source)],
                check=True, capture_output=True, timeout=60)
            for skipped, expected in ((False, 33), (True, 32)):
                out = base / f"synthetic_test_skip_{skipped}.mp4"
                transcode_one(source, out, skipped)
                v = [x for x in ffprobe(out)["streams"]
                     if x["codec_type"] == "video"][0]
                self.assertEqual(v["width"], 576)
                self.assertEqual(v["height"], 1024)
                self.assertEqual(v["avg_frame_rate"], "24/1")
                self.assertEqual(int(v["nb_read_frames"]), expected)

    def test_six_independent_voice_inputs_produce_full_length_audio_clock(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            wavs = {}
            for i, sid in enumerate(["S01", "S02", "S03", "S04", "S05", "S06"]):
                wave = base / f"{sid}.wav"
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y",
                     "-f", "lavfi", "-i", f"sine=frequency={300+i*65}:sample_rate=48000",
                     "-t", "0.15", "-ac", "1", "-c:a", "pcm_s16le", str(wave)],
                    capture_output=True, check=True, timeout=30)
                wavs[sid] = wave
            merged = base / "SYNTHETIC_TEST_NOT_REAL_RUSSIAN_AUDIO.wav"
            audio_mix(base, wavs, plan(), merged)
            streams = [x for x in ffprobe(merged)["streams"]
                       if x["codec_type"] == "audio"]
            self.assertEqual(len(streams), 1)
            self.assertAlmostEqual(float(streams[0]["duration"]),
                                   plan()["final_duration_seconds"], delta=0.04)

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
            self.assertNotIn("minterpolate", f)
            self.assertNotIn("fps=", f)
            self.assertNotIn("loop", f)

    def test_final_40_second_frame_math_is_only_a_plan(self):
        p = plan()
        self.assertEqual(p["new_true_inference_segment_count"], 30)
        self.assertEqual(p["final_unique_native_frames"], 967)
        self.assertEqual(p["original_russian_qwen_lines"], 6)
        self.assertFalse(p["production_executed"])
        self.assertFalse(p["cinematic_owner_approved"])


if __name__ == "__main__":
    unittest.main()
