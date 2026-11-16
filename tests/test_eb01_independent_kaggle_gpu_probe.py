"""No-network GPU scheduling proof for two real independent Kaggle tokens."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import eb01_independent_kaggle_gpu_probe as mod


class PrivateGPUProbeTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)

    def test_private_microkernel_has_zero_media_and_neural_model(self):
        self.assertIn('"private_actor_media_loaded":False', mod.CODE)
        self.assertIn('"neural_model_loaded":False', mod.CODE)
        self.assertNotIn("KAGGLE_API_TOKEN", mod.CODE)
        self.assertNotIn("KOSTYA.png", mod.CODE)
        self.assertNotIn("ELIZABETH.png", mod.CODE)

    def test_real_gpu_report_accepts_per_gpu_but_never_pools_memory(self):
        commands = []
        def fake_cmd(args,token,timeout=180):
            commands.append((args,token))
            if args[1:3] == ["kernels","push"]:
                kernel = Path(args[args.index("-p")+1])
                cfg=json.loads((kernel/"kernel-metadata.json").read_text())
                self.assertEqual(cfg["is_private"],"true")
                self.assertEqual(cfg["enable_internet"],"false")
                self.assertEqual(cfg["dataset_sources"],[])
                self.assertTrue(cfg["id"].startswith("tester/eb01-hw-probe-"))
            if args[1:3] == ["kernels","status"]:
                return "KernelWorkerStatus.COMPLETE"
            if args[1:3] == ["kernels","output"]:
                dest=Path(args[args.index("-p")+1])
                dest.joinpath("EB01_HARDWARE_ONLY_SAFE.json").write_text(
                    json.dumps({"status":"REAL_GPU_MEASURED",
                                "cuda_available":True,
                                "gpu_device_count":2,
                                "per_gpu_gib":[
                                    {"index":0,"physical_gib":14.56,"free_gib":13.2},
                                    {"index":1,"physical_gib":14.56,"free_gib":13.1}]}))
            return "OK"
        with patch.object(mod,"query",return_value=("AUTHENTICATED_READ_ONLY","tester")), \
             patch.object(mod,"safe_call",side_effect=fake_cmd):
            result=mod.probe_alias("collaborator","only-collab-token",
                                   "43901011111",self.root)
        self.assertEqual(result["status"],
                         "REAL_PRIVATE_GPU_HARDWARE_VERIFIED_NO_VIDEO")
        self.assertEqual(result["gpu_device_count"],2)
        self.assertEqual(result["gpu_submission_attempts"],1)
        self.assertTrue(result["gpu_measured"])
        self.assertFalse(result["repo_source_media_copied"])
        self.assertFalse(result["gpu_movie_generation"])
        self.assertEqual(len(commands),3)
        self.assertTrue(all(token=="only-collab-token" for args,token in commands))

    def test_source_account_username_never_appears_in_output(self):
        def failed(args,token,timeout=180):
            raise mod.GPUProbeBlocked("KAGGLE_GPU_QUOTA_EXHAUSTED")
        with patch.object(mod,"query",return_value=("AUTHENTICATED_READ_ONLY","hiddenusername")), \
             patch.object(mod,"safe_call",side_effect=failed):
            result=mod.probe_alias("owner","owner-token","43901011111",self.root)
        self.assertEqual(result["status"],"BLOCKED_KAGGLE_GPU_QUOTA_EXHAUSTED")
        self.assertNotIn("hiddenusername",str(result))
        self.assertEqual(result["gpu_submission_attempts"],1)
        self.assertFalse(result["gpu_measured"])

    def test_missing_account_rejected_without_inference(self):
        with patch.object(mod,"query",return_value=("MISSING_GITHUB_SECRET",None)), \
             patch.object(mod,"safe_call") as submit:
            result=mod.probe_alias("owner","","43901011111",self.root)
        self.assertEqual(result["status"],
                         "BLOCKED_AUTHENTICATED_KAGGLE_ACCOUNT_NOT_FOUND")
        self.assertEqual(result["gpu_submission_attempts"],0)
        submit.assert_not_called()

    def test_each_alias_uses_independent_stable_kernel_identifier(self):
        import hashlib
        def expected(alias):
            return hashlib.sha256(("eb01-hw-43901011111-"+alias).encode()).hexdigest()[:12]
        self.assertNotEqual(expected("owner"),expected("collaborator"))
        self.assertEqual(len(expected("owner")),12)


if __name__ == "__main__":
    unittest.main()
