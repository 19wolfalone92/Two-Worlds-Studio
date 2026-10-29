"""Offline Kaggle API tests. No tokens, GPU, Kaggle calls or model downloads."""
from __future__ import annotations
import json
import tempfile
import unittest
import zipfile
import io
from PIL import Image
from pathlib import Path
from types import SimpleNamespace

from studio.kaggle_api import (
    ROOT, SLUG, build_job, package_source, parse_status,
    poll_kernel, verify_outputs
)


class KaggleApiTests(unittest.TestCase):
    def test_reviewed_source_bundle_excludes_credentials(self):
        blob = package_source()
        self.assertGreater(len(blob), 1000)
        self.assertLess(len(blob), 1_500_000)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"modules.zip"
            path.write_bytes(blob)
            with zipfile.ZipFile(path) as archive:
                names=archive.namelist()
                self.assertIn("studio/kaggle_flux.py",names)
                self.assertIn("studio/kokoro_tts.py",names)
                self.assertFalse(any(".env" in p or ".github" in p for p in names))

    def test_private_gpu_kernel_is_prepared_without_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)
            info=build_job("examplekaggle","gh-102-1",output_dir=target,git_sha="a"*40)
            metadata=json.loads((target/"kernel-metadata.json").read_text())
            script=(target/"kernel_main.py").read_text()
            compile(script,"kernel_main.py","exec")
            self.assertEqual(info["kernel"],"examplekaggle/"+SLUG)
            self.assertTrue(info["source_bundle_bytes"]>1000)
            self.assertEqual(metadata["is_private"],"true")
            self.assertEqual(metadata["enable_gpu"],"true")
            self.assertEqual(metadata["machine_shape"],"NvidiaTeslaT4")
            self.assertEqual(metadata["kernel_type"],"script")
            self.assertNotIn("__TW_CONFIG_B64__",script)
            self.assertNotIn("__TW_BUNDLE_B64__",script)
            self.assertNotIn("KAGGLE_API_TOKEN",script)
            self.assertFalse(info["credential_included"])

    def test_voice_only_kernel_does_not_request_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            info=build_job("examplekaggle","gh-voice-1",mode="voice",
                           output_dir=Path(tmp))
            self.assertEqual(info["metadata"]["enable_gpu"],"false")
            self.assertEqual(info["metadata"]["machine_shape"],"")

    def test_input_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_job("../../wrong","valid-17",output_dir=Path(tmp))
            with self.assertRaises(ValueError):
                build_job("normaluser","../wrong",output_dir=Path(tmp))
            with self.assertRaises(ValueError):
                build_job("normaluser","valid-17",body="some_person",
                          output_dir=Path(tmp))

    def test_lifecycle_is_explicit(self):
        self.assertEqual(parse_status('owner/kernel has status "KernelWorkerStatus.RUNNING"'),"RUNNING")
        self.assertEqual(parse_status('owner/kernel has status "KernelWorkerStatus.COMPLETE"'),"COMPLETE")
        self.assertEqual(parse_status('owner/kernel has status "KernelWorkerStatus.ERROR"'),"ERROR")
        with self.assertRaises(ValueError):
            parse_status("Work completed maybe?")

        def success(cmd,**kwargs):
            self.assertEqual(cmd[:3],["kaggle","kernels","status"])
            return SimpleNamespace(stdout='name has status "KernelWorkerStatus.COMPLETE"')
        self.assertEqual(poll_kernel("person/"+SLUG,runner=success),"COMPLETE")

        def failure(cmd,**kwargs):
            return SimpleNamespace(stdout='name has status "KernelWorkerStatus.ERROR"')
        with self.assertRaisesRegex(RuntimeError,"ERROR"):
            poll_kernel("person/"+SLUG,runner=failure)

    def test_download_checks_run_nonce_and_real_media_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            (d/"two_worlds_job_receipt.json").write_text(
                json.dumps({"run_nonce":"gh-104-1","success":True}))
            archive=d/"TWO_WORLDS_ARTIFACTS.zip"
            pixels=io.BytesIO()
            Image.new("RGB",(64,64),(60,100,160)).save(pixels,format="PNG")
            with zipfile.ZipFile(archive,"w") as z:
                z.writestr("portrait.png",pixels.getvalue())
            self.assertTrue(verify_outputs(d,"gh-104-1")["success"])
            with zipfile.ZipFile(archive,"w") as z:
                z.writestr("portrait.png",b"not valid image bytes")
            with self.assertRaisesRegex(ValueError,"actual generated PNG"):
                verify_outputs(d,"gh-104-1")
            with self.assertRaisesRegex(ValueError,"another run"):
                verify_outputs(d,"gh-104-2")
            with zipfile.ZipFile(archive,"w") as z:
                z.writestr("log.txt","No images")
            with self.assertRaisesRegex(ValueError,"did not generate"):
                verify_outputs(d,"gh-104-1")

    def test_dispatch_manual_and_no_public_art_upload(self):
        yaml=(ROOT/".github/workflows/kaggle-api.yml").read_text()
        self.assertIn("workflow_dispatch:",yaml)
        self.assertNotIn("schedule:",yaml)
        self.assertIn("secrets.KAGGLE_API_TOKEN",yaml)
        self.assertIn("vars.KAGGLE_USERNAME",yaml)
        self.assertIn("github.event.repository.private",yaml)
        self.assertIn("kaggle kernels push",yaml)
        self.assertNotIn('echo "$KAGGLE_API_TOKEN"',yaml)
        self.assertNotIn("metricool",yaml.lower())


if __name__=="__main__":
    unittest.main()
