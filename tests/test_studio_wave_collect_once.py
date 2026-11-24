"""Mocked remote Kaggle status + protected readback chain, no GPU required."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from tools.studio_wave_collect_once import collect_once
from tools.studio_wave_status_gate import JobStateError

class WaveReadbackTests(unittest.TestCase):
    def test_gpu_queue_does_not_resubmit_download_or_claim_ready(self):
        for state in ("QUEUED","RUNNING"):
            with self.subTest(state=state),tempfile.TemporaryDirectory() as d:
                output=Mock(side_effect=AssertionError("output requested before COMPLETE"))
                verify=Mock(side_effect=AssertionError("verification before COMPLETE"))
                encrypt=Mock(side_effect=AssertionError("encryption before COMPLETE"))
                result=collect_once(Path(d),status_fn=lambda:f'KernelWorkerStatus.{state}',
                                    output_fn=output,verify_fn=verify,seal_fn=encrypt)
                self.assertEqual(result["status"],"WAITING_GPU" if state=="QUEUED" else "GPU_RUNNING")
                self.assertFalse(result["real_mp4_accepted"])
                self.assertFalse(result["safe_to_requeue"])
                self.assertEqual(result["new_gpu_jobs_submitted"],0)
                output.assert_not_called();verify.assert_not_called();encrypt.assert_not_called()

    def test_terminal_job_error_preserved_without_fake_video(self):
        with tempfile.TemporaryDirectory() as d:
            result=collect_once(Path(d),status_fn=lambda:'KernelWorkerStatus.ERROR',
                                output_fn=Mock(side_effect=AssertionError("no output")))
            self.assertEqual(result["status"],"KAGGLE_TERMINAL_FAILURE")
            self.assertTrue(result["requires_private_failure_forensic"])
            self.assertFalse(result["model_failure_confirmed"])
            self.assertFalse(result["real_mp4_accepted"])

    def test_remote_unknown_state_fails_not_pending_or_complete(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(JobStateError):
                collect_once(Path(d),status_fn=lambda:'KernelWorkerStatus.MAGIC')

    def test_complete_with_missing_media_fails_closed_without_encrypt(self):
        with tempfile.TemporaryDirectory() as d:
            seal=Mock()
            with self.assertRaisesRegex(JobStateError,"missing"):
                collect_once(Path(d),status_fn=lambda:'KernelWorkerStatus.COMPLETE',
                             output_fn=lambda _:None,seal_fn=seal)
            seal.assert_not_called()

    def test_real_video_verified_then_encrypted_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            verified=Mock(return_value={
                "status":"TRUE_NEURAL_MP4_TECHNICALLY_VERIFIED_OWNER_REVIEW_REQUIRED",
                "human_artistic_approved":False,"frames":65,"sha256":"a"*64})
            def output_fn(folder):
                self.assertTrue(folder.is_dir())
                (folder/"STUDIO_WAVE_TRUE_NATIVE_65F_24FPS.mp4").write_bytes(b"synthetic-test-fixture")
            ciphertext=Mock(return_value={"sha256":"b"*64})
            result=collect_once(root,status_fn=lambda:'KernelWorkerStatus.COMPLETE',
                                output_fn=output_fn,verify_fn=verified,seal_fn=ciphertext)
            self.assertEqual(result["status"],"TRUE_GPU_MP4_VERIFIED_ENCRYPTED_OWNER_REVIEW")
            self.assertTrue(result["private_result_encrypted"])
            self.assertFalse(result["human_artistic_approved"])
            self.assertEqual(result["new_gpu_jobs_submitted"],0)
            ciphertext.assert_called_once()
            self.assertTrue(str(ciphertext.call_args.args[0]).endswith("STUDIO_WAVE_TRUE_NATIVE_65F_24FPS.mp4"))

    def test_verifier_cannot_claim_approved_without_proof(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(JobStateError,"verification gate"):
                collect_once(Path(d),status_fn=lambda:'KernelWorkerStatus.COMPLETE',
                             output_fn=lambda _:None,
                             verify_fn=lambda _:{"status":"fake","human_artistic_approved":True},
                             seal_fn=Mock())

if __name__=="__main__":unittest.main()
