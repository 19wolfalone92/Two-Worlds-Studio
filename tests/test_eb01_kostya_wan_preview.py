"""Offline proof that skeleton I2V handoff only reads verified private canon art."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from studio.eb01_canon_kaggle import ARCHIVE as CANON_ARCHIVE, RECEIPT as CANON_RECEIPT
from studio.eb01_kostya_wan_preview import parent_reference, prepare


class E01PrivateWanProofTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        original=Image.effect_noise((704,1216),35).convert("RGB")
        import io
        raw=io.BytesIO();original.save(raw,format="PNG")
        self.source=raw.getvalue()
        self.source_sha=hashlib.sha256(self.source).hexdigest()
        with zipfile.ZipFile(self.root/CANON_ARCHIVE,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("canon/KOSTYA.png",self.source)
            z.writestr("canon/ELIZABETH.png",self.source)
        archive=self.root/CANON_ARCHIVE
        data={
            "success":True,"private":True,"gpu_inference_performed":True,
            "published":False,"model":"stabilityai/stable-diffusion-xl-base-1.0",
            "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
            "generated_images":{"KOSTYA":{"sha256":self.source_sha}}}
        (self.root/CANON_RECEIPT).write_text(json.dumps(data))

    def test_read_verified_real_private_parent(self):
        resized,sha=parent_reference(self.root)
        self.assertEqual(sha,self.source_sha)
        from PIL import Image
        import io
        with Image.open(io.BytesIO(resized)) as image:
            self.assertEqual(image.size,(448,800))

    def test_archive_tamper_blocks_gpu_preparation(self):
        payload=json.loads((self.root/CANON_RECEIPT).read_text())
        payload["archive_sha256"]="0"*64
        (self.root/CANON_RECEIPT).write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,"archive corrupt"):
            parent_reference(self.root)

    def test_only_private_worker_embeds_source_at_runtime(self):
        with tempfile.TemporaryDirectory() as scratch:
            based=Path(scratch)
            (based/"tools").mkdir()
            # Test placeholders mirror the proven original worker and are
            # replaced only within the temporary PRIVATE Kaggle job script.
            (based/"tools"/"kaggle_wan22_video_entry.py").write_text(
                'CONFIG="__TW_VIDEO_CONFIG__"\n'
                'BUNDLE="__TW_VIDEO_BUNDLE__"\n'
                'def run_job(data,reference):\n'
                '    draw_fixture(reference)\n'
                '    if data.get("frames") == 17: pass\n')
            with patch("studio.eb01_kostya_wan_preview._bundle",return_value=b"staged-source"):
                data=prepare("wolf19loner92","pilot-local-test-1",self.root,
                             based/"private-worker",root=based)
            self.assertEqual(data["frame_count"],49)
            self.assertFalse(data["gpu_executed"])
            self.assertEqual(data["original_canon_sha256"],self.source_sha)
            script=(based/"private-worker"/"kernel_main.py").read_text()
            self.assertNotIn("draw_fixture(reference)",script)
            self.assertNotIn("__TW_VIDEO_CONFIG__",script)
            self.assertIn("reference_png_b64",script)
            self.assertIn('data.get("frames") == 49',script)
            meta=json.loads((based/"private-worker"/"kernel-metadata.json").read_text())
            self.assertEqual(meta["is_private"],"true")


if __name__=="__main__":
    unittest.main()
