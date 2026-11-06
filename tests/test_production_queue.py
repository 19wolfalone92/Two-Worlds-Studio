"""Durable multi-provider CPU scheduling tests, no HTTP/GPU, no billing."""
from __future__ import annotations
import hashlib
import tempfile
import unittest
from pathlib import Path
from studio.production_queue import open_queue,submit,claim,succeed,fail,status,enqueue_routes

H="a"*64

class QueueTests(unittest.TestCase):
    def test_persistent_idempotency_and_payload_lock(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"studio.sqlite"
            with open_queue(path) as db:
                self.assertTrue(submit(db,"episode01.scene01","image_generation",H,now=100)["enqueued"])
                self.assertFalse(submit(db,"episode01.scene01","image_generation",H,now=101)["enqueued"])
                with self.assertRaises(ValueError):
                    submit(db,"episode01.scene01","image_generation","b"*64,now=102)
            with open_queue(path) as db:
                self.assertEqual(status(db)["jobs"][0]["count"],1)
                self.assertEqual(claim(db,"worker-01",now=110)["attempt"],1)
                self.assertIsNone(claim(db,"worker-02",now=111))
                self.assertEqual(succeed(db,"episode01.scene01","worker-01","b"*64,now=112)["status"],"complete")
                self.assertIsNone(claim(db,"worker-02",now=113))

    def test_rate_limit_backoff_retry_without_token_rotation(self):
        with tempfile.TemporaryDirectory() as td:
            with open_queue(Path(td)/"q.db") as db:
                submit(db,"episode01.scene02","image_generation",H,now=100)
                self.assertEqual(claim(db,"worker-a",now=110)["attempt"],1)
                retry=fail(db,"episode01.scene02","worker-a","http_429",now=120,retry_after=200)
                self.assertEqual(retry["retry_after_seconds"],200)
                self.assertIsNone(claim(db,"worker-b",now=319))
                self.assertEqual(claim(db,"worker-b",now=320)["attempt"],2)
                retry=fail(db,"episode01.scene02","worker-b","provider_unavailable",now=321)
                self.assertEqual(retry["retry_after_seconds"],120)
                self.assertIsNone(claim(db,"worker-c",now=440))
                self.assertEqual(claim(db,"worker-c",now=441)["attempt"],3)

    def test_expired_lease_recovery_and_terminal_errors(self):
        with tempfile.TemporaryDirectory() as td:
            with open_queue(Path(td)/"q.db") as db:
                submit(db,"ep01.sc03","russian_tts",H,now=0)
                claim(db,"worker-a",now=5,lease_seconds=30)
                with self.assertRaises(ValueError):
                    succeed(db,"ep01.sc03","worker-b","b"*64,now=6)
                recovered=claim(db,"worker-b",now=36,lease_seconds=30)
                self.assertEqual(recovered["attempt"],2)
                with self.assertRaises(ValueError):
                    succeed(db,"ep01.sc03","worker-a","b"*64,now=40)
                self.assertEqual(fail(db,"ep01.sc03","worker-b","unsafe_input",now=45)["status"],"failed")
                self.assertIsNone(claim(db,"worker-c",now=4000))

    def test_no_raw_error_messages_or_private_source_files(self):
        with tempfile.TemporaryDirectory() as td:
            with open_queue(Path(td)/"q.db") as db:
                with self.assertRaises(ValueError):
                    submit(db,"badid","network","x"*64)
                with self.assertRaises(ValueError):
                    submit(db,"ep01.sc04","montage","invalid hash")
                submit(db,"ep01.sc04","montage",H,now=0)
                claim(db,"worker-a",now=1)
                with self.assertRaises(ValueError):
                    fail(db,"ep01.sc04","worker-a","secret: API key 123",now=2)
                self.assertTrue(status(db)["no_raw_prompts_or_tokens"])
                self.assertFalse(status(db)["provider_execution_implemented"])

    def test_free_route_plan_can_be_staged_without_inference(self):
        from studio.free_studio import prepare_batch
        plan = prepare_batch(
            {"schema":1,"episode":"E01","scenes":[
                {"scene":"S01","prompt":"Real abandoned corridor cinematic atmosphere","seed":10},
                {"scene":"S02","prompt":"A lonely building at night with cinematic lighting","seed":11}
            ]},
            {"CLOUDFLARE_AI_TOKEN":"test","CLOUDFLARE_ACCOUNT_ID":"e"*32})
        with tempfile.TemporaryDirectory() as td:
            with open_queue(Path(td)/"q.db") as db:
                receipt=enqueue_routes(db,plan,now=100)
                self.assertEqual(receipt["new_jobs"],2)
                self.assertEqual(receipt["external_calls"],0)
                self.assertFalse(receipt["published"])
                self.assertEqual(enqueue_routes(db,plan,now=101)["new_jobs"],0)
                self.assertEqual(len(status(db)["jobs"]),1)
                self.assertEqual(status(db)["jobs"][0]["count"],2)

    def test_rate_limit_daily_quota_retry(self):
        with tempfile.TemporaryDirectory() as td:
            with open_queue(Path(td)/"q.db") as db:
                submit(db,"ep01.sc05","image_generation",H,now=0)
                claim(db,"worker-a",now=1)
                result=fail(db,"ep01.sc05","worker-a","quota_exhausted",now=2)
                self.assertGreaterEqual(result["retry_after_seconds"],86400)
                self.assertIsNone(claim(db,"worker-b",now=3600))

    def test_public_repository_database_blocked(self):
        from studio.production_queue import ROOT
        with self.assertRaisesRegex(ValueError,"public source repository"):
            open_queue(ROOT/"dist"/"tasks.sqlite")

if __name__=="__main__":
    unittest.main()
