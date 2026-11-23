"""Freevideo regression tests. No network, Kaggle, GPU or model downloads."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tw_t4_experiment", HERE / "adaptive_pipeline.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def digest(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def fake_preview(root: Path, *, prompt: str, seed: int = 42):
    ref, clip, review = root / "ref.png", root / "preview.mp4", root / "review.json"
    ref.write_bytes(b"real project reference placeholder")
    clip.write_bytes(b"synthetic movie bytes; _video_probe is mocked in policy tests")
    clip_hash = digest(clip)
    source_hash = digest(ref)
    bridge_path = clip.with_suffix(".receipt.json")
    bridge_path.write_text(json.dumps({
        "schema": 1, "engine": "neural_comfy", "model": "wan22-5b",
        "clip_sha256": clip_hash, "reference_sha256": source_hash,
        "inference_performed": True, "human_approved": False, "published": False,
        "workflow_sha256": "a" * 64, "seed": seed,
        "video": {"frames": 25, "width": 432, "height": 768, "fps": "24"}}))
    clip.with_suffix(".freevideo.json").write_text(json.dumps({
        "schema": 2, "stage": "preview", "backend": "wan22-5b",
        "clip_sha256": clip_hash, "reference_sha256": source_hash,
        "bridge_receipt_sha256": digest(bridge_path),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "human_approved": False, "seed": seed,
        "geometry": {"width": 432, "height": 768, "frames": 25, "steps": 20},
    }))
    review.write_text(json.dumps({"schema": 1, "approved": True, "reviewer": "Owner",
                                  "preview_sha256": clip_hash, "reference_sha256": source_hash,
                                  "checks": dict.fromkeys(module.CHECKS, True)}))
    return ref, clip, review


class PlanningTests(unittest.TestCase):
    def test_t4_plan_incompatible_h3_bf16_and_frame_geometry(self):
        p = module.choose_plan(module.simulate_t4())
        self.assertEqual(p["status"], "HEURISTIC_PLAN_NOT_BENCHMARKED")
        self.assertEqual(p["backend"], "wan22-5b")
        self.assertFalse(p["bf16_allowed"])
        self.assertIn("freevideo_h3_not_t4_compatible", p["excluded"])
        self.assertEqual(p["final"], {"width": 576, "height": 1024, "frames": 49, "steps": 36})
        for stage in ("preview", "final"):
            dims = p[stage]
            self.assertEqual(dims["width"] * 16, dims["height"] * 9)
            self.assertEqual((dims["frames"] - 1) % 4, 0)
            self.assertTrue(12 <= dims["steps"] <= 50)

    def test_low_memory_fails_closed(self):
        lower = module.Hardware("Tesla T4", "7.5", 16, 8, "detected_cuda")
        self.assertEqual(module.choose_plan(lower)["profile"], "low_vram_cautious")
        self.assertEqual(module.choose_plan(lower)["final"]["width"], 432)
        for h in [module.Hardware("Old", "6.1", 8, 8, "detected_cuda"),
                  module.Hardware("T4", "7.5", 16, 4, "detected_cuda"),
                  module.Hardware("T4", "7.5", 16, float("nan"), "detected_cuda"),
                  module.Hardware("T4", "7.5", 16, 18, "detected_cuda")]:
            with self.subTest(h=h), self.assertRaises(module.Blocked):
                module.choose_plan(h)

    def test_simulation_cannot_trigger_generation_or_gpu_probe(self):
        with patch.object(module, "detect_hardware", side_effect=AssertionError("GPU probe")):
            self.assertEqual(module.main(["plan", "--simulate-t4"]), 0)
            self.assertEqual(module.main(["preview", "--simulate-t4"]), 2)

    def test_authorization_gate_precedes_gpu_probe(self):
        args = argparse.Namespace(command="preview", execute=False, accept_model_license=False)
        with patch.object(module, "detect_hardware", side_effect=AssertionError("GPU probe")):
            with self.assertRaises(module.Blocked):
                module.run(args)

    def test_single_worker_device_lock_prevents_concurrent_execution(self):
        with module.device_lease(0):
            with self.assertRaisesRegex(module.Blocked, "busy"):
                with module.device_lease(0):
                    self.fail("Lock must reject another job on the same device")
        # A normally completed context releases the per-device file lock.
        with module.device_lease(0):
            pass
        with self.assertRaises(module.Blocked):
            with module.device_lease(-1):
                pass

    def test_busy_gpu_lock_blocks_before_cuda_or_bridge(self):
        args = argparse.Namespace(command="preview", execute=True,
                                 accept_model_license=True, device=0)
        with module.device_lease(0):
            with patch.object(module, "detect_hardware") as gpu, patch.object(
                 module.subprocess, "run") as bridge:
                with self.assertRaisesRegex(module.Blocked, "busy"):
                    module.run(args)
                gpu.assert_not_called()
                bridge.assert_not_called()

    def test_t4_only_for_real_execution(self):
        args = argparse.Namespace(command="preview", execute=True, accept_model_license=True, device=0)
        with patch.object(module, "detect_hardware", return_value=module.Hardware(
                "RTX 4090", "8.9", 24, 20, "detected_cuda")):
            with self.assertRaisesRegex(module.Blocked, "only a real NVIDIA Tesla T4"):
                module.run(args)


class SecurityAndProvenanceTests(unittest.TestCase):
    def test_loopback_only_host(self):
        module._check_host("http://127.0.0.1:8188")
        for invalid in ("https://127.0.0.1:8188", "http://192.168.1.10:8188",
                        "http://localhost", "http://a:b@127.0.0.1:8188",
                        "http://127.0.0.1:8188/other", "http://127.0.0.1:8188/?x=1"):
            with self.subTest(invalid=invalid), self.assertRaises(module.Blocked):
                module._check_host(invalid)

    def test_comfy_device_consistency(self):
        h = module.Hardware("Tesla T4", "7.5", 16, 14, "detected_cuda", 0)
        class Response:
            def __init__(self, payload):
                self.headers = {"Content-Length": str(len(payload))}
                self.payload = payload
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def read(self, *_): return self.payload
        class Opener:
            def __init__(self, doc): self.doc = doc
            def open(self, *_args, **_kwargs): return Response(json.dumps(self.doc).encode())
        with patch.object(module, "build_opener", return_value=Opener({
            "devices": [{"type": "cuda", "index": 0, "name": "cuda:0 Tesla T4", "vram_total": 16 * 2**30}]})):
            self.assertIn("CONSISTENT", module.attest_backend("http://127.0.0.1:8188", h)["status"])
        for bad in [{"devices": []},
                    {"devices": [{"type": "cuda", "index": 1, "name": "Tesla T4", "vram_total": 16 * 2**30}]},
                    {"devices": [{"type": "cuda", "index": 0, "name": "RTX 4090", "vram_total": 24 * 2**30}]},
                    {"devices": [{"type": "cuda", "index": 0, "name": "Tesla T4", "vram_total": 8 * 2**30}]}]:
            with self.subTest(bad=bad), patch.object(module, "build_opener", return_value=Opener(bad)):
                with self.assertRaises(module.Blocked):
                    module.attest_backend("http://127.0.0.1:8188", h)

    def test_private_no_public_path_symlink_or_alias(self):
        with self.assertRaises(module.Blocked):
            module._private_path(module.REPO_ROOT / "experiments" / "bad.mp4", must_exist=False)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = root / "actual.json"
            real.write_text("{}")
            link = root / "alias.json"
            link.symlink_to(real)
            with self.assertRaises(module.Blocked):
                module._private_path(link)
            with self.assertRaises(module.Blocked):
                module._history_append(link, {"stage": "preview"})

    def test_human_gate_with_both_receipts_and_video_clock(self):
        prompt = "A skeletal character turns slowly without bone deformation."
        with tempfile.TemporaryDirectory() as temp:
            ref, clip, review = fake_preview(Path(temp), prompt=prompt)
            verified_video = {"width": 432, "height": 768, "frames": 25, "fps": "24"}
            with patch.object(module, "_video_probe", return_value=verified_video):
                self.assertTrue(module.validate_review(review, clip, ref, prompt=prompt, seed=42)["approved"])
                with self.assertRaisesRegex(module.Blocked, "prompt differs"):
                    module.validate_review(review, clip, ref, prompt=prompt + " new", seed=42)
                with self.assertRaisesRegex(module.Blocked, "seed differs"):
                    module.validate_review(review, clip, ref, prompt=prompt, seed=101)
                content = json.loads(review.read_text())
                content["checks"]["anatomy"] = False
                review.write_text(json.dumps(content))
                with self.assertRaisesRegex(module.Blocked, "manually approved"):
                    module.validate_review(review, clip, ref, prompt=prompt, seed=42)

    def test_missing_or_inconsistent_receipts_fail_closed(self):
        prompt = "A figure moves through a slow natural head turn."
        with tempfile.TemporaryDirectory() as temp:
            ref, clip, review = fake_preview(Path(temp), prompt=prompt)
            bridge = clip.with_suffix(".receipt.json")
            stage = clip.with_suffix(".freevideo.json")
            with patch.object(module, "_video_probe", return_value={
                    "width": 432, "height": 768, "frames": 25, "fps": "24"}):
                bridge.unlink()
                with self.assertRaises(module.Blocked):
                    module.validate_review(review, clip, ref, prompt=prompt, seed=42)
                bridge.write_text(json.dumps({"clip_sha256": digest(clip), "inference_performed": False}))
                with self.assertRaises(module.Blocked):
                    module.validate_review(review, clip, ref, prompt=prompt, seed=42)
                bridge.unlink()
                stage.unlink()
                with self.assertRaises(module.Blocked):
                    module.validate_review(review, clip, ref, prompt=prompt, seed=42)

    def test_corrupt_or_malformed_review_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "r.json"
            path.write_text("[]")
            with self.assertRaises(module.Blocked):
                module._json_file(path)
            path.write_text("{" + "x" * 10)
            with self.assertRaises(module.Blocked):
                module._json_file(path)
            path.write_text("x" * 32769)
            with self.assertRaises(module.Blocked):
                module._json_file(path)

    def test_ffprobe_real_video_and_invalid_bytes(self):
        if not module.subprocess.run(["sh", "-c", "command -v ffmpeg && command -v ffprobe"],
                                      capture_output=True).returncode == 0:
            self.skipTest("ffmpeg/ffprobe not installed")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "good.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi",
                            "-i", "color=c=black:s=432x768:r=24", "-frames:v", "25",
                            "-c:v", "mpeg4", "-qscale:v", "20", str(path)],
                           stdout=subprocess.DEVNULL, check=True)
            self.assertEqual(module._video_probe(path), {
                "width": 432, "height": 768, "fps": "24", "frames": 25})
            path.write_bytes(b"a photo, not a movie")
            with self.assertRaises(module.Blocked):
                module._video_probe(path)


class ExecutionAndTelemetryTests(unittest.TestCase):
    def _args(self, root: Path, reference: Path, *, stage="preview", prompt=None):
        # Three MOCKED checkpoint-byte fixtures exercise the hash gate only.
        # They do not emulate Wan model inference or verify true model origin.
        from weights_preflight import REQUIRED
        weights = {}
        for filename in REQUIRED:
            path = root / filename
            path.write_bytes(("MOCKED WEIGHTS " + filename).encode())
            weights[filename] = {"path": str(path), "sha256": digest(path)}
        manifest = root / "mock_checkpoint_manifest.json"
        manifest.write_text(json.dumps({
            "schema": 1, "model": "wan22-5b",
            "license_reviewed": True,
            "source_repository": "https://huggingface.co/Comfy-Org/",
            "files": weights}))
        return argparse.Namespace(command=stage, execute=True, accept_model_license=True,
            device=0, reference=str(reference), output=str(root / ("take_" + stage + ".mp4")),
            prompt=prompt or "A human character turns their shoulders naturally.",
            seed=42, timeout=3600, host="http://127.0.0.1:8188",
            history=str(root / "timing.jsonl"), review=None, preview_video=None,
            weights_manifest=str(manifest))

    def _bridge(self, command, **kwargs):
        self.assertEqual(command[0:3], [sys.executable, "-m", "studio.comfy_neural"])
        self.assertIn("--execute-local-gpu", command)
        self.assertIn("--accept-model-license", command)
        output = Path(command[command.index("--output") + 1])
        ref = Path(command[command.index("--reference") + 1])
        output.write_bytes(b"fake output from mocked subprocess (not real GPU)")
        d = digest(output)
        output.with_suffix(".receipt.json").write_text(json.dumps({
            "schema": 1, "engine": "neural_comfy", "clip_sha256": d,
            "reference_sha256": digest(ref), "model": "wan22-5b",
            "inference_performed": True, "human_approved": False, "published": False,
            "seed": int(command[command.index("--seed") + 1]),
            "workflow_sha256": "a" * 64,
            "video": {"width": int(command[command.index("--width") + 1]),
                      "height": int(command[command.index("--height") + 1]),
                      "frames": int(command[command.index("--frames") + 1]),
                      "fps": "24"}}))
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "status": "GENERATED_REVIEW_REQUIRED", "human_approved": False,
            "clip_sha256": d}), "")

    def test_mocked_bridge_provenance_stats_and_no_publishing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref = root / "source.png"
            ref.write_bytes(b"reference")
            args = self._args(root, ref)
            hardware = module.Hardware("Tesla T4", "7.5", 16, 14, "detected_cuda")
            with patch.object(module, "detect_hardware", return_value=hardware), patch.object(
                 module, "attest_backend", return_value={"status": "mock"}), patch.object(
                 module, "_video_probe", return_value={"width": 432, "height": 768,
                                                       "fps": "24", "frames": 25}), patch.object(
                 module.subprocess, "run", side_effect=self._bridge):
                result = module.run(args)
            self.assertEqual(result["status"], "GENERATED_REVIEW_REQUIRED")
            self.assertFalse(result["published"])
            self.assertFalse(result["human_approved"])
            self.assertTrue(Path(result["stage_receipt"]).exists())
            self.assertEqual(module.summary(Path(args.history))["successful_runs"], 1)
            self.assertEqual(json.loads(Path(result["stage_receipt"]).read_text())["stage"], "preview")

    def test_failed_history_append_does_not_erase_confirmed_generated_video(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref = root / "ref.png"; ref.write_bytes(b"reference")
            args = self._args(root, ref)
            with patch.object(module, "detect_hardware", return_value=module.Hardware(
                  "Tesla T4", "7.5", 16, 14, "detected_cuda")), patch.object(
                  module, "attest_backend", return_value={"status": "mock"}), patch.object(
                  module, "_video_probe", return_value={"width": 432, "height": 768,
                                                         "fps": "24", "frames": 25}), patch.object(
                  module.subprocess, "run", side_effect=self._bridge), patch.object(
                  module, "_history_append", side_effect=OSError("Disk full")):
                result = module.run(args)
            self.assertEqual(result["status"], "GENERATED_REVIEW_REQUIRED")
            self.assertIn("telemetry could not be saved", result["history_warning"])
            self.assertTrue(Path(args.output).exists())

    def test_final_review_check_blocks_subprocess_even_with_real_t4(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref = root / "ref.png"; ref.write_bytes(b"reference")
            args = self._args(root, ref, stage="final")
            with patch.object(module, "detect_hardware", return_value=module.Hardware(
                  "Tesla T4", "7.5", 16, 14, "detected_cuda")), patch.object(
                  module, "attest_backend") as attestation, patch.object(
                  module.subprocess, "run") as actual:
                with self.assertRaisesRegex(module.Blocked, "Final requires"):
                    module.run(args)
                actual.assert_not_called()
                attestation.assert_not_called()

    def test_invalid_subprocess_json_blocks_without_claiming_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref = root / "ref.png"; ref.write_bytes(b"reference")
            args = self._args(root, ref)
            with patch.object(module, "detect_hardware", return_value=module.Hardware(
                  "Tesla T4", "7.5", 16, 14, "detected_cuda")), patch.object(
                  module, "attest_backend", return_value={"status": "mock"}), patch.object(
                  module.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "[]", "")):
                with self.assertRaisesRegex(module.Blocked, "verified, unapproved"):
                    module.run(args)

    def test_complete_two_stage_handoff_with_separate_receipts(self):
        """Only simulated *bridge* operations; proves wrapper's chain, not GPU."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "ref.png"
            reference.write_bytes(b"reference for mocked two-stage test")
            prompt = "The skeletal character moves shoulders naturally with a rigid skull."
            hardware = module.Hardware("Tesla T4", "7.5", 16, 14, "detected_cuda")
            def decoded_video(file):
                if Path(file).name == "take_preview.mp4":
                    return {"width": 432, "height": 768, "frames": 25, "fps": "24"}
                return {"width": 576, "height": 1024, "frames": 49, "fps": "24"}
            preview = self._args(root, reference, prompt=prompt)
            final = self._args(root, reference, stage="final", prompt=prompt)
            with patch.object(module, "detect_hardware", return_value=hardware), patch.object(
                 module, "attest_backend", return_value={"status": "mock"}), patch.object(
                 module, "_video_probe", side_effect=decoded_video), patch.object(
                 module.subprocess, "run", side_effect=self._bridge) as called:
                preview_result = module.run(preview)
                self.assertEqual(called.call_count, 1)
                review_path = root / "manual_review.json"
                review_path.write_text(json.dumps({
                    "schema": 1, "approved": True, "reviewer": "Human reviewer",
                    "preview_sha256": preview_result["clip_sha256"],
                    "reference_sha256": digest(reference),
                    "checks": dict.fromkeys(module.CHECKS, True)}))
                final.review = str(review_path)
                final.preview_video = preview.output
                final_result = module.run(final)
                self.assertEqual(called.call_count, 2)
            self.assertEqual(final_result["status"], "GENERATED_REVIEW_REQUIRED")
            self.assertFalse(final_result["human_approved"])
            final_proof = json.loads(Path(final_result["stage_receipt"]).read_text())
            self.assertEqual(final_proof["stage"], "final")
            self.assertEqual(final_proof["geometry"]["frames"], 49)
            self.assertEqual(module.summary(Path(final.history))["successful_runs"], 2)

    def test_immutable_output_prevents_duplicate_gpu_spend(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref = root / "ref.png"; ref.write_bytes(b"reference")
            args = self._args(root, ref)
            Path(args.output).write_bytes(b"do not overwrite")
            with patch.object(module, "detect_hardware", return_value=module.Hardware(
                  "Tesla T4", "7.5", 16, 14, "detected_cuda")), patch.object(
                  module, "attest_backend") as attest, patch.object(
                  module.subprocess, "run") as process:
                with self.assertRaisesRegex(module.Blocked, "new private MP4"):
                    module.run(args)
                process.assert_not_called()
                attest.assert_not_called()
            self.assertEqual(Path(args.output).read_bytes(), b"do not overwrite")

    def test_preview_seed_tamper_prevents_final_without_spending_gpu(self):
        prompt = "A skeletal character rotates its head without face warping."
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference, preview, review = fake_preview(root, prompt=prompt)
            bridge = preview.with_suffix(".receipt.json")
            info = json.loads(bridge.read_text())
            info["seed"] = 999
            bridge.write_text(json.dumps(info))
            proof = preview.with_suffix(".freevideo.json")
            stage = json.loads(proof.read_text())
            stage["bridge_receipt_sha256"] = digest(bridge)
            proof.write_text(json.dumps(stage))
            with patch.object(module, "_video_probe", return_value={
                    "width": 432, "height": 768, "frames": 25, "fps": "24"}):
                with self.assertRaisesRegex(module.Blocked, "Studio seed differs"):
                    module.validate_review(review, preview, reference, prompt=prompt, seed=42)

    def test_successful_measurements_only_and_no_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "runs.jsonl"
            for elapsed in (10.0, 12.0, 20.0):
                module._history_append(file, {"result": "GENERATED_REVIEW_REQUIRED", "backend": "wan22-5b",
                         "hardware": "Tesla T4", "stage": "preview", "resolution": "432x768",
                         "frames": 25, "elapsed_s": elapsed})
            self.assertEqual(module.summary(file)["groups"][0]["median_elapsed_s"], 12)
            for value in (float("nan"), -1, "oops"):
                with self.subTest(value=value):
                    file.write_text(json.dumps({"result": "GENERATED_REVIEW_REQUIRED", "backend": "wan22-5b",
                          "hardware": "Tesla T4", "stage": "preview", "resolution": "432x768",
                          "frames": 25, "elapsed_s": value}))
                    with self.assertRaises(module.Blocked):
                        module.summary(file)

    def test_unknown_metrics_lines_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "m.jsonl"
            for bad in ("[]\n", "{not json}", '{"result":"FAILED"}\n', "x" * 5000):
                p.write_text(bad)
                with self.subTest(bad=bad[:20]), self.assertRaises(module.Blocked):
                    module.summary(p)


    def test_changed_reference_during_gpu_is_not_approved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "ref.png"
            reference.write_bytes(b"original reference")
            args = self._args(root, reference)
            def mutate_source(command, **kwargs):
                finished = self._bridge(command, **kwargs)
                reference.write_bytes(b"unexpected replacement during render")
                return finished
            with patch.object(module, "detect_hardware", return_value=module.Hardware(
                    "Tesla T4", "7.5", 16, 14, "detected_cuda")), patch.object(
                    module, "attest_backend", return_value={"status": "mock"}), patch.object(
                    module.subprocess, "run", side_effect=mutate_source):
                with self.assertRaisesRegex(module.Blocked, "Reference image changed"):
                    module.run(args)

    def test_bridge_receipt_frame_clock_tampering_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "ref.png"
            reference.write_bytes(b"unchanged reference")
            args = self._args(root, reference)
            def tamper_bridge(command, **kwargs):
                finished = self._bridge(command, **kwargs)
                output = Path(command[command.index("--output") + 1])
                receipt = output.with_suffix(".receipt.json")
                doc = json.loads(receipt.read_text())
                doc["video"]["frames"] = 999
                receipt.write_text(json.dumps(doc))
                return finished
            with patch.object(module, "detect_hardware", return_value=module.Hardware(
                    "Tesla T4", "7.5", 16, 14, "detected_cuda")), patch.object(
                    module, "attest_backend", return_value={"status": "mock"}), patch.object(
                    module, "_video_probe", return_value={"width": 432, "height": 768,
                                                        "fps": "24", "frames": 25}), patch.object(
                    module.subprocess, "run", side_effect=tamper_bridge):
                with self.assertRaisesRegex(module.Blocked, "Studio receipt"):
                    module.run(args)


    def test_wrong_weight_digest_blocks_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "ref.png"
            reference.write_bytes(b"untouched reference")
            args = self._args(root, reference)
            manifest = Path(args.weights_manifest)
            doc = json.loads(manifest.read_text())
            component = next(iter(doc["files"].values()))
            component["sha256"] = "0" * 64
            manifest.write_text(json.dumps(doc))
            with patch.object(module, "detect_hardware", return_value=module.Hardware(
                    "Tesla T4", "7.5", 16, 14, "detected_cuda")), patch.object(
                    module, "attest_backend") as attestation, patch.object(
                    module.subprocess, "run") as bridge:
                with self.assertRaisesRegex(module.Blocked, "integrity preflight"):
                    module.run(args)
                bridge.assert_not_called()
                attestation.assert_not_called()

    def test_missing_private_weight_manifest_blocks_gpu_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "ref.png"
            reference.write_bytes(b"untouched reference")
            args = self._args(root, reference)
            args.weights_manifest = None
            with patch.object(module, "detect_hardware", return_value=module.Hardware(
                    "Tesla T4", "7.5", 16, 14, "detected_cuda")), patch.object(
                    module, "attest_backend") as attestation, patch.object(
                    module.subprocess, "run") as bridge:
                with self.assertRaisesRegex(module.Blocked, "--weights-manifest"):
                    module.run(args)
                bridge.assert_not_called()
                attestation.assert_not_called()


    def test_gpu_unknown_timeout_persists_quarantine_and_blocks_repeat(self):
        """MOCKED timeout must fail closed without re-submitting to ComfyUI."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref = root / "ref.png"
            ref.write_bytes(b"approved placeholder, not a real portrait")
            args = self._args(root, ref)
            # Choose a distinct test GPU index: no real CUDA device is accessed.
            args.device = 31
            marker = module._quarantine_marker(31)
            marker.unlink(missing_ok=True)
            try:
                mocked_t4 = module.Hardware("Tesla T4", "7.5", 16, 14, "detected_cuda", 31)
                with patch.object(module, "detect_hardware", return_value=mocked_t4), patch.object(
                     module, "attest_backend", return_value={"status": "MOCKED"}), patch.object(
                     module.subprocess, "run",
                     side_effect=subprocess.TimeoutExpired(cmd="MOCKED bridge", timeout=1)) as bridge:
                    with self.assertRaisesRegex(module.Blocked, "GPU_QUARANTINED"):
                        module.run(args)
                    self.assertEqual(bridge.call_count, 1)
                    self.assertTrue(marker.is_file())
                    proof = json.loads(marker.read_text())
                    self.assertEqual(proof["status"], "GPU_QUARANTINED")
                    with self.assertRaisesRegex(module.Blocked, "GPU_QUARANTINED"):
                        module.run(args)
                    self.assertEqual(bridge.call_count, 1, "No second GPU command is allowed")
            finally:
                marker.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
