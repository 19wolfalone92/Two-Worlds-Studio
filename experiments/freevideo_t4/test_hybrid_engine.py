"""End-to-end offline boundaries for the isolated Freevideo + Studio hybrid.

Any mocked inference below is explicitly MOCKED; genuine GPU tests require
an actual Kaggle T4 and locally installed model weights.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hybrid_engine as hy


class HybridEngineTests(unittest.TestCase):
    def source(self, root: Path, *, size=(600, 800)) -> Path:
        path = root / "photo.png"
        Image.new("RGB", size, (51, 102, 153)).save(path)
        return path

    def prepare(self, root: Path):
        src = self.source(root)
        work = root / "private-job"
        report = hy.prepare(src, work, "night-awakens-001",
                            "A seated subject takes a slow natural breath and turns head.",
                            seed=77)
        return src, work, report

    def test_prepare_original_not_touched_and_whole_image_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self.source(root)
            digest_before = hy.sha256(src)
            report = hy.prepare(src, root / "private-job", "night-awakens-001",
                                "A person is gently breathing in a fixed room.", seed=3)
            self.assertEqual(report["status"], "PREPARED_NO_GPU")
            self.assertEqual(hy.sha256(src), digest_before)
            manifest = json.loads(Path(report["job_file"]).read_text())
            self.assertEqual(manifest["source_size"], [600, 800])
            self.assertEqual(manifest["reference_size"], [864, 1536])
            self.assertEqual(manifest["resized_content_size"], [864, 1152])
            self.assertEqual(manifest["content_origin"], [0, 192])
            self.assertEqual(manifest["framing"], "contain_without_cropping")
            with Image.open(root / "private-job" / "reference_9x16.png") as final:
                self.assertEqual(final.size, (864, 1536))
                self.assertEqual(final.getpixel((430, 500)), (51, 102, 153))
            self.assertEqual(hy.status(report["job_file"])["takes"]["preview"]["status"], "NOT_GENERATED")

    def test_native_9x16_reference_keeps_whole_picture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self.source(root, size=(540, 960))
            doc = hy.prepare(src, root / "job", "photo-001",
                             "A person blinks, moves hands gently, and smiles.")
            record = json.loads(Path(doc["job_file"]).read_text())
            self.assertEqual(record["resized_content_size"], [864, 1536])
            self.assertEqual(record["content_origin"], [0, 0])

    def test_existing_private_job_and_original_are_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, work, doc = self.prepare(root)
            with self.assertRaisesRegex(hy.HybridBlocked, "NEW private"):
                hy.prepare(src, work, "night-awakens-001",
                           "A subject takes a quiet breath while camera is fixed.")
            self.assertTrue(src.is_file())
            self.assertTrue((work / "job.json").is_file())

    def test_corrupted_photo_rejected_without_creating_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "bad.png"
            src.write_bytes(b"fake image header")
            with self.assertRaises(hy.HybridBlocked):
                hy.prepare(src, root / "job", "photo-bad",
                           "A person breathes and opens their eyes.")
            self.assertFalse((root / "job").exists())

    def test_modified_original_or_reference_blocks_entire_pipeline(self):
        for which in ("source", "reference"):
            with self.subTest(which=which), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                src, work, doc = self.prepare(root)
                target = src if which == "source" else work / "reference_9x16.png"
                target.write_bytes(b"changed")
                with self.assertRaisesRegex(hy.HybridBlocked, "changed"):
                    hy.status(doc["job_file"])

    def test_unapproved_no_gpu_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, work, doc = self.prepare(root)
            with patch.object(hy.adaptive_pipeline, "run") as bridge:
                with self.assertRaisesRegex(hy.HybridBlocked, "explicit execution"):
                    hy.run_stage(doc["job_file"], "preview", weights_manifest="x")
                with self.assertRaisesRegex(hy.HybridBlocked, "weights-manifest"):
                    hy.run_stage(doc["job_file"], "preview",
                                 execute=True, accept_model_license=True)
                with self.assertRaisesRegex(hy.HybridBlocked, "Human-reviewed"):
                    hy.run_stage(doc["job_file"], "final",
                                 execute=True, accept_model_license=True, weights_manifest="x")
                bridge.assert_not_called()

    def test_no_fake_automatic_human_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, work, doc = self.prepare(root)
            with self.assertRaisesRegex(hy.HybridBlocked, "Actual generated preview"):
                hy.review_template(doc["job_file"])
            self.assertFalse((work / "human_review_TEMPLATE.json").exists())

    def test_preview_mocked_bridge_passes_only_approved_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, work, doc = self.prepare(root)
            def fake_inference(arguments):
                self.assertEqual(arguments.command, "preview")
                self.assertEqual(arguments.seed, 77)
                self.assertEqual(arguments.reference, str(work / "reference_9x16.png"))
                self.assertTrue(arguments.accept_model_license)
                self.assertTrue(arguments.execute)
                self.assertEqual(arguments.device, 0)
                self.assertEqual(arguments.weights_manifest, "/private/weights.json")
                # Synthetic data here: intentionally NOT a playable MP4.
                output = Path(arguments.output)
                output.write_bytes(b"MOCKED NOT NEURAL VIDEO")
                for suffix in (".receipt.json", ".freevideo.json"):
                    output.with_suffix(suffix).write_text('{"status":"MOCKED"}')
                return {"status": "GENERATED_REVIEW_REQUIRED", "clip_sha256": hy.sha256(output)}
            with patch.object(hy.adaptive_pipeline, "run", side_effect=fake_inference), patch(
                    "studio.video_qc.sample_video",
                    return_value={"frames_sampled": 25, "automated_aesthetic_approval": False}):
                r = hy.run_stage(doc["job_file"], "preview", execute=True,
                                 accept_model_license=True, weights_manifest="/private/weights.json")
            self.assertEqual(r["status"], "GENERATED_REVIEW_REQUIRED")
            self.assertEqual(r["visual_quality"], "NOT_VERIFIED")
            proof = json.loads((work / "preview.hybrid.json").read_text())
            self.assertEqual(proof["neural_inference"], "GPU_BRIDGE_REPORTED")
            self.assertFalse(proof["human_approved"])
            self.assertEqual(proof["model_path"], "studio.comfy_neural")
            self.assertEqual(hy.status(doc["job_file"])["takes"]["preview"]["status"],
                             "GENERATED_REVIEW_REQUIRED")
            with patch.object(hy.adaptive_pipeline, "run") as bridge:
                with self.assertRaisesRegex(hy.HybridBlocked, "never re-render"):
                    hy.run_stage(doc["job_file"], "preview", execute=True,
                                 accept_model_license=True, weights_manifest="/private/weights.json")
                bridge.assert_not_called()

    def test_review_template_must_default_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, work, doc = self.prepare(root)
            clip = work / "preview.mp4"
            clip.write_bytes(b"MOCKED CLIP, ONLY PROVENANCE CONTRACT")
            d = hy.sha256(clip)
            (work / "preview.receipt.json").write_text(json.dumps({"clip_sha256": d}))
            manifest = json.loads((work / "job.json").read_text())
            (work / "preview.hybrid.json").write_text(json.dumps({
                "clip_sha256": d,
                "reference_sha256": manifest["reference_sha256"],
                "source_sha256": manifest["source_sha256"]}))
            out = hy.review_template(doc["job_file"])
            data = json.loads(Path(out["template"]).read_text())
            self.assertEqual(out["status"], "VISUAL_REVIEW_REQUIRED")
            self.assertFalse(data["approved"])
            self.assertFalse(any(data["checks"].values()))
            with self.assertRaisesRegex(hy.HybridBlocked, "overwrite"):
                hy.review_template(doc["job_file"])

    def test_tampered_stage_sha_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, work, doc = self.prepare(root)
            (work / "preview.mp4").write_bytes(b"MOCKED VIDEO")
            (work / "preview.hybrid.json").write_text(json.dumps({
                "job_id": "night-awakens-001", "stage": "preview",
                "clip_sha256": "0"*64, "visual_quality": "NOT_VERIFIED",
                "human_approved": False}))
            with self.assertRaisesRegex(hy.HybridBlocked, "tampered"):
                hy.status(doc["job_file"])

    def test_malformed_prompt_seed_and_job_id_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = self.source(root)
            for job_id, prompt, seed in [
                ("../../studio", "A person turns slowly and naturally.", 42),
                ("valid-123", "short", 42),
                ("valid-123", "A person turns slowly and naturally.", -1)
            ]:
                with self.subTest(job_id=job_id, seed=seed):
                    with self.assertRaises(hy.HybridBlocked):
                        hy.prepare(src, root / "job", job_id, prompt, seed=seed)
                    self.assertFalse((root / "job").exists())

    def test_duplicate_json_keys_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, work, doc = self.prepare(root)
            p = work / "job.json"
            p.write_text('{"schema":1,"schema":1}')
            with self.assertRaisesRegex(hy.HybridBlocked, "Duplicate JSON"):
                hy.status(doc["job_file"])


if __name__ == "__main__":
    unittest.main()
