"""Zero GPU regression for SaveKernel 400, protected original Kostya, single Elizabeth source."""
import base64
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from studio.eb01_canon_kaggle import (
    ARCHIVE as PARENT_ARCHIVE, CONTACT as PARENT_CONTACT,
    RECEIPT as PARENT_RECEIPT, MODEL as PARENT_MODEL
)
from studio.eb01_elizabeth_photoreal_repair import (
    ARCHIVE, CONTACT, RECEIPT, MODEL, MAX_NOTEBOOK_SOURCE_BYTES,
    PARENT_KERNEL, original_kostya, prepare, verify
)


class RepairOnlyElizabethTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root / "parent"
        self.out = self.root / "repaired"
        self.parent.mkdir()
        self.out.mkdir()
        self.parent_nonce = "eb01canon-test-nonce"
        self.nonce = "eb01repair-test-nonce"
        # Test-only noise is NOT a valid artistic video frame or identity proof.
        self.kostya = self.image_bytes(704, 1216)
        self.original_sha = hashlib.sha256(self.kostya).hexdigest()
        self.create_parent()

    def image_bytes(self, w, h):
        buf = io.BytesIO()
        Image.effect_noise((w, h), 37).convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    def create_parent(self):
        liz = self.image_bytes(704, 1216)
        with zipfile.ZipFile(self.parent / PARENT_ARCHIVE, "w") as z:
            z.writestr("canon/KOSTYA.png", self.kostya)
            z.writestr("canon/ELIZABETH.png", liz)
        Image.effect_noise((484, 483), 33).convert("RGB").save(
            self.parent / PARENT_CONTACT, format="JPEG")
        d = {
            "success": True, "nonce": self.parent_nonce, "model": PARENT_MODEL,
            "private": True, "gpu_inference_performed": True,
            "video_generated": False, "published": False,
            "image_identity_verified": False, "full_video_allowed": False,
            "archive_sha256": hashlib.sha256(
                (self.parent / PARENT_ARCHIVE).read_bytes()).hexdigest(),
            "contact_sha256": hashlib.sha256(
                (self.parent / PARENT_CONTACT).read_bytes()).hexdigest(),
            "generated_images": {
                who: {"sha256": hashlib.sha256(raw).hexdigest(),
                      "size": len(raw), "width": 704, "height": 1216,
                      "art_reviewed": False}
                for who, raw in (("KOSTYA", self.kostya), ("ELIZABETH", liz))
            }
        }
        (self.parent / PARENT_RECEIPT).write_text(json.dumps(d))

    def create_photo_output(self):
        photo = self.image_bytes(704, 1216)
        with zipfile.ZipFile(self.out / ARCHIVE, "w") as z:
            z.writestr("canon/ELIZABETH.png", photo)
        Image.effect_noise((272, 520), 37).convert("RGB").save(
            self.out / CONTACT, format="JPEG")
        d = {
            "schema": 2, "nonce": self.nonce,
            "success": True, "private": True, "model": MODEL,
            "parent_KOSTYA_sha256": self.original_sha,
            "parent_source_kernel": PARENT_KERNEL,
            "original_KOSTYA_untouched": True,
            "gpu_inference_performed": True, "image_identity_verified": False,
            "full_video_allowed": False, "video_generated": False,
            "published": False,
            "archive_sha256": hashlib.sha256(
                (self.out / ARCHIVE).read_bytes()).hexdigest(),
            "preview_sha256": hashlib.sha256(
                (self.out / CONTACT).read_bytes()).hexdigest(),
            "generated_images": {
                "ELIZABETH": {
                    "sha256": hashlib.sha256(photo).hexdigest(),
                    "size": len(photo), "width": 704, "height": 1216,
                    "art_reviewed": False, "model": MODEL
                }
            }
        }
        (self.out / RECEIPT).write_text(json.dumps(d))
        return d

    def test_original_skeleton_reused_by_sha_only(self):
        raw, sha = original_kostya(self.parent, self.parent_nonce)
        self.assertEqual(raw, self.kostya)
        self.assertEqual(sha, self.original_sha)

    def test_real_job_code_is_small_and_contains_no_original(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root)
            (source / "tools").mkdir()
            (source / "tools" / "kaggle_eb01_elizabeth_repair_entry.py").write_text(
                'CONFIG_B64 = "__EB01_ELIZABETH_REPAIR_CONFIG_B64__"\n')
            result = prepare(
                "wolf19loner92", self.nonce, self.parent, self.parent_nonce,
                source / "private_job", root=source)
            self.assertEqual(result["new_images"], 1)
            self.assertTrue(result["private_original_source_unchanged"])
            self.assertFalse(result["binary_image_embedded_in_notebook"])
            self.assertLess(result["submitted_notebook_bytes"],
                            MAX_NOTEBOOK_SOURCE_BYTES)
            worker = (source / "private_job" / "kernel_main.py").read_text()
            self.assertNotIn(self.original_sha[:12] + "==", worker)
            self.assertNotIn("KAGGLE_API_TOKEN=", worker)
            self.assertNotIn("iVBOR", worker)  # PNG Base64 prefix
            self.assertNotIn("kostya_png_b64", worker.lower())
            token = worker.split('"')[1]
            decoded = json.loads(base64.b64decode(token).decode())
            self.assertEqual(decoded["parent_KOSTYA_sha256"], self.original_sha)
            self.assertEqual(decoded["max_new_images"], 1)

    def test_verified_result_has_only_elizabeth_photo(self):
        self.create_photo_output()
        report = verify(self.out, self.nonce, self.original_sha)
        self.assertEqual(report["unchanged_KOSTYA_SHA256"], self.original_sha)
        self.assertFalse(report["visual_identity_approved"])
        self.assertFalse(report["published"])
        with zipfile.ZipFile(self.out / ARCHIVE) as z:
            self.assertEqual(z.namelist(), ["canon/ELIZABETH.png"])

    def test_wrong_parent_skeleton_blocks_approval(self):
        self.create_photo_output()
        with self.assertRaisesRegex(ValueError, "provenance"):
            verify(self.out, self.nonce, "a"*64)

    def test_tampering_single_original_blocks_approval(self):
        data = self.create_photo_output()
        data["generated_images"]["ELIZABETH"]["sha256"] = "b"*64
        (self.out / RECEIPT).write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            verify(self.out, self.nonce, self.original_sha)


if __name__ == "__main__":
    unittest.main()
