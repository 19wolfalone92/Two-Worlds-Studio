"""Tests for 3-part real media handoff and release quality gates.

No network, Kaggle GPU, external API or publishing required.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from studio.release_gate import cover_check, inspect_subtitles, check_video, opening_three_seconds, sha256
from studio.triptych_batch import read_package_manifest, make_bundle


class ReleaseQA(unittest.TestCase):
    def test_russian_srt_and_timing(self):
        with tempfile.TemporaryDirectory() as td:
            file = Path(td) / "test.srt"
            file.write_text("1\n00:00:00,000 --> 00:00:01,000\nЭто начало.\n\n"
                            "2\n00:00:01,000 --> 00:00:02,000\nВторая строка.\n",
                            encoding="utf-8")
            self.assertEqual(inspect_subtitles(file, 2)["cues"], 2)
            with self.assertRaisesRegex(ValueError, "beyond"):
                inspect_subtitles(file, 1)
            file.write_text("1\n00:00:00,500 --> 00:00:00,900\nЭто начало.\n\n"
                            "2\n00:00:00,400 --> 00:00:01,000\nВторая строка.\n")
            with self.assertRaisesRegex(ValueError, "overlapping"):
                inspect_subtitles(file, 2)

    def test_color_cover_and_black_cover(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            cover = d / "cover.png"
            Image.new("RGB", (540, 960), (30, 55, 170)).save(cover)
            self.assertTrue(cover_check(cover)["first_frame_valid"])
            Image.new("RGB", (540, 960), "black").save(cover)
            with self.assertRaisesRegex(ValueError, "cover"):
                cover_check(cover)

    def test_opening_three_second_black_frame_gate(self):
        from types import SimpleNamespace
        size = 96 * 170
        fully_black = bytes(size) * 4
        with patch("studio.release_gate.subprocess.run",
                   return_value=SimpleNamespace(stdout=fully_black)):
            with self.assertRaisesRegex(ValueError, "sustained black"):
                opening_three_seconds(Path("mock.mp4"))
        mostly_visible = bytes([55]) * size * 5 + bytes([120]) * size
        with patch("studio.release_gate.subprocess.run",
                   return_value=SimpleNamespace(stdout=mostly_visible)):
            response = opening_three_seconds(Path("mock.mp4"))
            self.assertEqual(response["frames_sampled"], 6)
            self.assertEqual(response["black_frames"], 0)

    def test_check_video_requires_true_h264_aac_30_fps(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            for f in ("test.mp4", "test.srt", "cover.png"):
                (p / f).write_bytes(b"test")
            probe = {
                "format": {"duration": "4.0"},
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "width": 1080,
                     "height": 1920, "avg_frame_rate": "30/1"},
                    {"codec_type": "audio", "codec_name": "aac"}
                ]
            }
            with patch("studio.release_gate.ffprobe", return_value=probe), \
                 patch("studio.release_gate.inspect_subtitles", return_value={"cues": 1}), \
                 patch("studio.release_gate.cover_check", return_value={"first_frame_valid": True}), \
                 patch("studio.release_gate.first_frame_check", return_value={"first_frame_valid": True}), \
                 patch("studio.release_gate.opening_three_seconds", return_value={"frames_sampled": 6}):
                self.assertEqual(check_video(p / "test.mp4", p / "test.srt", p / "cover.png")["status"], "technical_pass")
                probe["streams"][0]["avg_frame_rate"] = "24/1"
                with self.assertRaisesRegex(ValueError, "30 FPS"):
                    check_video(p / "test.mp4", p / "test.srt", p / "cover.png")


def wav(path: Path) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\x00\x00" * 6400)


def fake_production(root: Path) -> Path:
    src = root / "images"
    src.mkdir()
    img = Image.new("RGB", (640, 1152), "#2d4965")
    ImageDraw.Draw(img).ellipse((110, 170, 520, 720), fill="#ddba91")
    img.save(src / "scene.png")
    
    wav(src / "speech.wav")
    parts = []
    for i in range(1, 4):
        Image.new("RGB", (540, 960), (45+i*16, 64+i*20, 100+i*9)).save(src / f"cover{i}.png")
        content = {
            "schema": 1, "width": 1080, "height": 1920, "fps": 30,
            "burn_subtitles": True,
            "shots": [
                {
                    "image": "images/scene.png", "duration": 1.5,
                    "camera": "push_in", "caption": "Слова у двери.",
                    "voice": "images/speech.wav",
                } for _ in range(8)
            ],
        }
        path = root / f"part{i}.json"
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
        parts.append({"part": i, "manifest": path.name, "cover": f"images/cover{i}.png",
                      "expected_shots": 8})
    spec = root / "production.json"
    spec.write_text(json.dumps({
        "schema": 1, "type": "three_part_story", "job_id": "ARCHIVE17_PILOT",
        "story_title": "ДЕВОЧКА НА ФОТОГРАФИИ", "language": "ru",
        "parts": parts,
    }, ensure_ascii=False), encoding="utf-8")
    return spec


class TriptychSafety(unittest.TestCase):
    def test_preflight_counts_all_24_real_shots(self):
        with tempfile.TemporaryDirectory() as td:
            spec = fake_production(Path(td))
            result = read_package_manifest(spec)
            self.assertEqual(result["shots"], 24)
            self.assertEqual([s["part"] for s in result["parts"]], [1, 2, 3])
            self.assertEqual([s["voice_tracks"] for s in result["parts"]], [8, 8, 8])
            with self.assertRaisesRegex(ValueError, "owner"):
                read_package_manifest(spec, master=True)

    def test_reject_wrong_order_and_escaping_source(self):
        with tempfile.TemporaryDirectory() as td:
            spec = fake_production(Path(td))
            raw = json.loads(spec.read_text(encoding="utf-8"))
            raw["parts"][0]["part"] = 2
            spec.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ordered"):
                read_package_manifest(spec)
            raw["parts"][0]["part"] = 1
            raw["parts"][0]["manifest"] = "../../unauthorized.json"
            spec.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escaping"):
                read_package_manifest(spec)

    def test_atomic_three_part_package_and_safe_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = fake_production(root)
            out = root / "dist"
            def fake_produce(manifest: Path, video: Path, master: bool = False):
                video.write_bytes(b"mock-videodata")
                video.with_suffix(".srt").write_text(
                    "1\n00:00:00,000 --> 00:00:01,000\nСлова у двери.\n", encoding="utf-8")
                video.with_suffix(".json").write_text('{"published": false}', encoding="utf-8")
            with patch("studio.triptych_batch.produce", side_effect=fake_produce) as produce_mock, \
                 patch("studio.triptych_batch.check_video", side_effect=lambda video, srt, cover, **_: {
                     "status": "technical_pass", "sha256": sha256(video),
                     "cover": {"sha256": sha256(cover)}}) as check_mock:
                result = make_bundle(spec, out)
                self.assertEqual(result["parts"], 3)
                self.assertFalse(result["published"])
                self.assertEqual(produce_mock.call_count, 3)
                target = Path(result["directory"])
                self.assertTrue((target / "delivery.json").is_file())
                self.assertTrue((target / "ARCHIVE17_PILOT_review_bundle.zip").is_file())
                self.assertEqual(make_bundle(spec, out)["status"], "cached_verified")
                self.assertEqual(produce_mock.call_count, 3)
                self.assertEqual(check_mock.call_count, 6)
                # A valid-looking but replaced output MP4 is NOT a valid cache hit.
                video = target / "PART_1_of_3.mp4"
                video.write_bytes(b"altered-after-export")
                with self.assertRaisesRegex(ValueError, "modified or corrupted"):
                    make_bundle(spec, out)
                video.write_bytes(b"mock-videodata")
                self.assertEqual(make_bundle(spec, out)["status"], "cached_verified")
                # Cache must NEVER reuse an old video after external art changes.
                Image.new("RGB", (540, 960), "#316bb2").save(
                    root / "images" / "cover1.png")
                with self.assertRaisesRegex(FileExistsError, "incomplete or has different"):
                    make_bundle(spec, out)

    def test_music_changes_invalidate_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = fake_production(root)
            music = root / "images" / "score.wav"
            wav(music)
            part = root / "part1.json"
            data = json.loads(part.read_text(encoding="utf-8"))
            data["music"] = "images/score.wav"
            part.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            before = read_package_manifest(spec)["source_assets_fingerprint"]
            with wave.open(str(music), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(16000)
                out.writeframes(b"\\x01\\x02" * 6400)
            after = read_package_manifest(spec)["source_assets_fingerprint"]
            self.assertNotEqual(before, after)

    def test_minimum_shot_count_cannot_be_lowered_by_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = fake_production(root)
            doc = json.loads(spec.read_text(encoding="utf-8"))
            doc["parts"][0]["expected_shots"] = 1
            spec.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected"):
                read_package_manifest(spec)

    def test_master_requires_actual_locked_work_screenplay(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec = fake_production(root)
            raw = json.loads(spec.read_text(encoding="utf-8"))
            raw["editorial_approved"] = True
            raw["rights_approved"] = True
            raw["final_human_review_approved"] = True
            raw["work_script_sha256"] = "0" * 64
            raw["work_script_file"] = "locked.md"
            (root / "locked.md").write_text("Work-approved script unchanged", encoding="utf-8")
            spec.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "locked SHA256"):
                read_package_manifest(spec, master=True)


if __name__ == "__main__":
    unittest.main()
