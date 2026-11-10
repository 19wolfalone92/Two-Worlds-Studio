import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageChops

from studio import neural_character as nc
from studio.hybrid_master import check_manifest, produce
from studio.release_gate import sha256
from studio.smooth_camera import encode_rgb_frames


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class CharacterTakeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.reference = self.root / "portrait.png"
        self.voice = self.root / "voice.wav"
        self.preview = self.root / "worker-fixture.mp4"
        self.width, self.height, self.fps = 288, 512, 24
        rng = np.random.default_rng(21)
        gray = rng.integers(35, 205, (self.height // 2, self.width // 2), dtype=np.uint8)
        self.plate = Image.fromarray(gray).convert("RGB").resize((self.width, self.height))
        self.plate.save(self.reference)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
                        "-i", "sine=f=213:d=0.75", "-ar", "24000", str(self.voice)], check=True)
        box = (111, 233, 171, 269)
        lip = self.plate.crop(box)

        def pictures():
            for i in range(41):
                frame = self.plate.copy()
                frame.paste(ImageChops.offset(lip, round(2.5 * np.sin(i * .3)), 0), box[:2])
                yield frame

        encode_rgb_frames(self.preview, self.width, self.height, 25, 41, pictures())
        self.pinned = {"path": self.reference.name, "sha256": sha256(self.reference)}
        self.windows = [{"start": .25, "end": 1.1}]

    def _regions(self):
        face = [.30, .24, .65, .43]
        wardrobe = [.23, .72, .73, .90]
        return [
            {"id": "camera", "kind": "camera", "box": [.12, .06, .85, .15]},
            {"id": "head", "kind": "face", "box": face,
             "identity_reference": {**self.pinned, "box": face}, "motion_windows": self.windows},
            {"id": "mouth", "kind": "mouth", "box": [.39, .46, .59, .522],
             "relative_to": "head", "motion_windows": self.windows},
            {"id": "jaw", "kind": "jaw", "box": [.3, .55, .64, .65],
             "relative_to": "head", "motion_windows": self.windows},
            {"id": "cheek", "kind": "cheek", "box": [.64, .45, .76, .545],
             "relative_to": "head", "motion_windows": self.windows},
            {"id": "shoulders", "kind": "wardrobe", "box": wardrobe,
             "identity_reference": {**self.pinned, "box": wardrobe}, "motion_windows": self.windows},
        ]

    def _manifest(self, take, *, echo=True):
        driver = self.voice if echo else self.preview
        shot = {
            "clip": "take/take.mp4", "duration": 1.5, "shot_type": "close" if echo else "reaction",
            "camera": "static", "approved": True,
            "clip_take": {"engine": "neural_character", "visual_profile": "photoreal_cinematic",
                          "sha256": take["clip_sha256"], "receipt": "take/take.json",
                          "reference": self.reference.name, "driver": driver.name},
            "qc_regions": self._regions(),
        }
        if echo:
            shot.update(voice=self.voice.name, voice_offset=.25, voice_approved=True,
                        dialogue_text="Кто там?", caption="Кто там?")
        else:
            shot["qc_regions"][1]["relative_to"] = "shoulders"
            shot["qc_regions"] = [r for r in shot["qc_regions"] if r["kind"] not in {"mouth", "jaw", "cheek"}]
        return {"schema": 1, "width": self.width, "height": self.height, "fps": self.fps,
                "visual_profile": "photoreal_cinematic", "editorial_approved": True,
                "rights_approved": True, "shots": [shot]}

    def _generate_echo(self):
        runtime, weights = self.root / "echo", self.root / "weights"
        (runtime / "config").mkdir(parents=True)
        (runtime / "config/config.yaml").write_text("test: true", encoding="utf-8")
        for folder in ("Wan2.1-Fun-V1.1-1.3B-InP", "chinese-wav2vec2-base", "transformer"):
            (weights / folder).mkdir(parents=True)
        checkpoint = weights / "transformer/diffusion_pytorch_model.safetensors"
        checkpoint.write_bytes(b"synthetic worker fixture")
        (runtime / "infer_flash.py").write_text(
            "from pathlib import Path\nimport shutil, sys, wave\n"
            "def arg(name): return sys.argv[sys.argv.index(name) + 1]\n"
            "with wave.open(arg('--audio_path'), 'rb') as wav:\n"
            "    assert wav.getframerate() == 16000 and wav.getnchannels() == 1\n"
            "    assert wav.readframes(4000) == bytes(8000)\n"
            "    assert any(wav.readframes(8000))\n"
            "assert int(arg('--video_length')) == 41 and arg('--fps') == '25'\n"
            f"shutil.copyfile({str(self.preview)!r}, Path(arg('--save_path')) / 'reference_output.mp4')\n",
            encoding="utf-8",
        )
        proof = patch.object(nc, "ECHO_CHECKPOINT_SHA", sha256(checkpoint))
        proof.start()
        self.addCleanup(proof.stop)
        return self._generate(nc.ECHO, runtime, weights, self.voice, .25)

    def _generate(self, model, runtime, weights, driver, offset=0):
        with patch.object(nc, "_cuda_dtype", return_value="bfloat16"), patch.object(
                nc, "_runtime_info", return_value={"revision": "a" * 40, "entry_sha256": "b" * 64}):
            return nc.generate_take(
                model, reference=self.reference, driver=driver, duration=1.5,
                voice_offset_ms=round(offset * 1000), seed=43,
                prompt="Synthetic test performance" if model == nc.ECHO else None,
                runtime=runtime, weights=weights, python=Path(sys.executable), output=self.root / "take",
            )

    def test_echo_take_preserves_wav_clock_and_rejects_stale_or_short_media(self):
        take = self._generate_echo()
        doc = self._manifest(take)
        shot = check_manifest(doc, self.root)["shots"][0]
        self.assertEqual(shot["motion_receipt"]["model"], nc.ECHO)
        self.assertTrue(shot["strict_clip"])
        self.assertEqual(shot["voice_offset"], .25)
        self.assertEqual(take["status"], "review_required")
        doc["shots"][0]["voice_offset"] = .3
        with self.assertRaisesRegex(ValueError, "stale"):
            check_manifest(doc, self.root)
        doc["shots"][0]["voice_offset"] = .25
        doc["shots"][0]["duration"] = 2
        with self.assertRaisesRegex(ValueError, "shorter than"):
            check_manifest(doc, self.root)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "cinematic extra required")
    def test_actual_mp4_with_only_lips_moving_cannot_pass_master(self):
        take = self._generate_echo()
        doc = self._manifest(take)
        manifest = self.root / "shot.json"
        manifest.write_text(json.dumps(doc), encoding="utf-8")
        preview = self.root / "review.mp4"
        result = produce(manifest, preview)
        self.assertTrue(result["qc"]["delivery"]["full_decode_pass"])
        flow = result["shot_reports"][0]["temporal_qc"]["optical_flow"]
        groups = {region["id"]: [event["code"] for event in region["events"]] for region in flow["regions"]}
        self.assertNotIn("no_independent_action", groups["mouth"])
        self.assertIn("no_independent_action", groups["jaw"])
        self.assertIn("no_independent_action", groups["cheek"])
        self.assertIn("no_independent_action", groups["shoulders"])
        self.assertEqual(result["qc"]["status"], "REVIEW REQUIRED")
        rejected = self.root / "approved.mp4"
        with self.assertRaisesRegex(ValueError, "unresolved motion or identity"):
            produce(manifest, rejected, master=True)
        self.assertFalse(rejected.exists())

    def test_wan_requires_a_driver_and_head_motion_separate_from_shoulders(self):
        runtime, weights = self.root / "wan", self.root / "wan-weights"
        preprocessor = runtime / nc.WAN_PREPROCESSOR
        preprocessor.parent.mkdir(parents=True)
        for filename in ("pose2d/vitpose_h_wholebody.onnx", "det/yolov10m.onnx"):
            model_file = weights / "process_checkpoint" / filename
            model_file.parent.mkdir(parents=True)
            model_file.write_bytes(b"synthetic fixture")
        (weights / "config.json").write_text("{}", encoding="utf-8")
        (weights / "diffusion_pytorch_model.safetensors.index.json").write_text("{}", encoding="utf-8")
        preprocessor.write_text(
            "import sys\nfrom pathlib import Path\n"
            "root = Path(sys.argv[sys.argv.index('--save_path') + 1])\n"
            "root.mkdir()\n"
            "for name in ('src_face.mp4', 'src_pose.mp4'):\n"
            "    (root / name).touch()\n",
            encoding="utf-8",
        )
        (runtime / "generate.py").write_text(
            "import shutil, sys\nfrom pathlib import Path\n"
            "assert Path(sys.argv[sys.argv.index('--src_root_path') + 1]).is_dir()\n"
            f"shutil.copyfile({str(self.preview)!r}, sys.argv[sys.argv.index('--save_file') + 1])\n",
            encoding="utf-8",
        )
        take = self._generate(nc.WAN, runtime, weights, self.preview)
        doc = self._manifest(take, echo=False)
        self.assertEqual(check_manifest(doc, self.root)["shots"][0]["motion_receipt"]["model"], nc.WAN)
        doc["shots"][0]["qc_regions"][1]["relative_to"] = "camera"
        with self.assertRaisesRegex(ValueError, "independently of shoulders"):
            check_manifest(doc, self.root)


if __name__ == "__main__":
    unittest.main()
