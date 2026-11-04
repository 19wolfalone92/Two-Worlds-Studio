"""Checks that unsafe eyelid overlays and cut-out sticker scenes cannot be masters."""
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw
from studio.actor_motion import animate_actor, validate_motion
from studio.animation_quality import audit_performance
from studio.handoff_audit import audit_handoff
from studio.work_contract import script_fingerprint


class StudioFirstQualityTests(unittest.TestCase):
    def test_unmasked_blink_is_frozen_not_superimposed_on_open_eye(self):
        base=Image.new("RGBA",(100,100),(180,135,90,255))
        patch=Image.new("RGBA",(100,100),(0,0,0,0))
        ImageDraw.Draw(base).rectangle((30,39,70,54),fill=(5,20,30,255))
        ImageDraw.Draw(patch).rectangle((30,43,70,47),fill=(20,20,20,255))
        config={"blink":{"at":[.5],"duration":.1}}
        no_mask=animate_actor(base,config,.5,overlay=patch)
        self.assertEqual(no_mask.getpixel((50,43)),base.getpixel((50,43)))
        erase=Image.new("L",(100,100),0)
        ImageDraw.Draw(erase).rectangle((30,39,70,54),fill=255)
        masked=animate_actor(base,config,.5,overlay=patch,blink_erase_mask=erase)
        self.assertEqual(masked.getpixel((50,39))[3],0)
        self.assertEqual(masked.getpixel((50,44)),(20,20,20,255))
        self.assertEqual(masked.getpixel((10,10)),base.getpixel((10,10)))

    def test_approved_blink_needs_hashed_aligned_erase_mask(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            overlay=root/"closed.png";mask=root/"erase.png"
            Image.new("RGBA",(90,140),(0,0,0,0)).save(overlay)
            Image.new("L",(90,140),0).save(mask)
            meta={"blink":{"at":[.5],"duration":.05,"path":"closed.png",
                  "review_status":"approved","rights":"owned",
                  "rights_note":"Art created by owner specifically for shot."}}
            self.assertEqual(validate_motion(meta,root)["blink_frames"],0)
            with self.assertRaisesRegex(ValueError,"eye-erasure mask"):
                validate_motion(meta,root,publish=True)
            meta["blink"]["erase_mask"]="erase.png"
            meta["blink"]["erase_mask_sha256"]=hashlib.sha256(mask.read_bytes()).hexdigest()
            self.assertTrue(validate_motion(meta,root,publish=True)["blink_safe"])

    def test_sticker_master_blocked_even_if_art_resolutions_are_large(self):
        episode={"scenes":[{"id":"S01","cinematic":{"layers":[{"role":"background"},
                    {"role":"character"}]}}]}
        r=audit_performance(episode)
        self.assertGreater(r["problems"],0)
        self.assertFalse(r["master_ready"])
        with self.assertRaisesRegex(ValueError,"not master-ready"):
            audit_performance(episode,publish=True)

    def test_authored_performance_must_be_reviewed_and_grounded(self):
        ep={"scenes":[{"id":"S01","cinematic":{"layers":[
            {"role":"background"},{"role":"character"}]},
            "animation":{"kind":"drawn_sequence","cues":["gaze","pose"],
            "integration":{"contact_shadow":True,"lighting":True,"perspective":True},
            "reviewed_by":"actual art director"}}]}
        result=audit_performance(ep,publish=True)
        self.assertTrue(result["master_ready"])
        ep["scenes"][0]["animation"]["blink_style"]="overlay_only"
        with self.assertRaisesRegex(ValueError,"blink overlay"):
            audit_performance(ep,publish=True)

    def test_private_handoff_audit_does_not_need_approved_script_for_preview(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"handoff.zip"
            image=Image.new("RGB",(300,500),(30,40,50))
            b=io.BytesIO();image.save(b,format="PNG")
            raw=b.getvalue()
            layer={"role":"background","path":"E01/shared/bg.png",
                   "sha256":hashlib.sha256(raw).hexdigest()}
            scene={"id":"S01","duration":1.0,"palette":"earth","effect":"none",
                   "heading":"TEST","caption":"Preview only",
                   "cinematic":{"layers":[layer]}}
            ep={"id":"E01","title":"Technical test","scenes":[scene]}
            ep["work_lock"]={"script_sha256":script_fingerprint(ep),
                            "editorial_approved":False}
            with zipfile.ZipFile(p,"w") as z:
                z.writestr("Studio/EP01_WORK_CANDIDATE_v1_1.json",json.dumps(ep))
                z.writestr("Studio/assets/E01/shared/bg.png",raw)
            result=audit_handoff(p)
            self.assertEqual(result["scenes"],1)
            self.assertTrue(result["work_script_hash_verified"])
            self.assertFalse(result["professional_animation_approved"])

if __name__ == "__main__":
    unittest.main()
