"""Safety validation: patch existing queued private notebook only, no secret leakage."""
from __future__ import annotations
import base64
import hashlib
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.studio_wave_patch_queued_private import (
    update_private_script,patch_queued_once,OLD,NEW,
)
from tools.studio_wave_status_gate import KERNEL,JobStateError

class PrivateQueuedPatchTests(unittest.TestCase):
    def fixture(self):
        actor=b"private-video-unit-test-source-NOT-A-USER-PHOTO"
        value=base64.b64encode(actor).decode()
        source=(
            'from studio.eb01_native_t4_profile import patch_native_sources\n'
            'IMAGE_B64="'+value+'"\n'
            'def main():\n'
            '    if native_profile:\n'
            +OLD+'\n'
        )
        return source, hashlib.sha256(actor).hexdigest(),value

    def test_patch_is_exact_and_original_photo_is_byte_identical(self):
        source,digest,value=self.fixture()
        corrected=update_private_script(source,image_sha=digest)
        self.assertIn(NEW,corrected)
        self.assertNotIn(OLD+"\n",corrected)
        self.assertIn('IMAGE_B64="'+value+'"',corrected)
        self.assertEqual(corrected.count('IMAGE_B64="'+value+'"'),1)
        with self.assertRaisesRegex(JobStateError,"already patched|known broken"):
            update_private_script(corrected,image_sha=digest)

    def test_wrong_photo_or_altered_revision_aborts_without_push(self):
        source,digest,_=self.fixture()
        with self.assertRaisesRegex(JobStateError,"differs"):
            update_private_script(source,image_sha="0"*64)
        with self.assertRaisesRegex(JobStateError,"exact known"):
            update_private_script(source.replace("eb01_native_t4_profile","invalid_source"),image_sha=digest)

    def test_running_and_complete_cannot_modify_kaggle(self):
        for state in ("RUNNING","COMPLETE","ERROR"):
            with self.subTest(state=state),tempfile.TemporaryDirectory() as d:
                calls=[]
                def fake_call(argv,timeout=200):
                    calls.append(argv)
                    return "KernelWorkerStatus."+state
                with patch("tools.studio_wave_patch_queued_private.run_cli",fake_call):
                    result=patch_queued_once(Path(d))
                self.assertFalse(result["submitted"])
                self.assertFalse(result["patched"])
                self.assertEqual(len(calls),1)
                self.assertEqual(calls[0][:3],["kaggle","kernels","status"])

    def test_gpu_allocation_race_aborts_before_push(self):
        source,digest,_=self.fixture()
        statuses=iter(["QUEUED","RUNNING"])
        calls=[]
        with tempfile.TemporaryDirectory() as d:
            def fake_call(argv,timeout=200):
                calls.append(argv)
                if argv[2]=="status":
                    return "KernelWorkerStatus."+next(statuses)
                if argv[2]=="pull":
                    folder=Path(argv[argv.index("-p")+1])
                    (folder/"kernel_main.py").write_text(source)
                    return "pulled"
                raise AssertionError("NO_PUSH_AFTER_RUNNING")
            with patch("tools.studio_wave_patch_queued_private.run_cli",fake_call),\
                 patch("tools.studio_wave_patch_queued_private.update_private_script",
                       lambda value: update_private_script(value,image_sha=digest)):
                result=patch_queued_once(Path(d))
            self.assertFalse(result["submitted"])
            self.assertEqual(result["reason"],"GPU_JOB_BECAME_ACTIVE_DURING_SAFE_PATCH")
            self.assertEqual([c[2] for c in calls],["status","pull","status"])

if __name__=="__main__":
    unittest.main()
