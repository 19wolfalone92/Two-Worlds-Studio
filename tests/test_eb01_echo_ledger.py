import tempfile,unittest
from pathlib import Path
from studio.eb01_echo_ledger import EchoLedger,QueueBlocked
class EchoLedgerTests(unittest.TestCase):
 def test_two_durable_queues_hold_disjoint_fifteen_jobs_and_no_generation(self):
  with tempfile.TemporaryDirectory() as t:
   allids=[]
   for alias in ('owner','collaborator'):
    q=EchoLedger(Path(t)/(alias+'.sqlite'),alias)
    self.assertEqual(q.summary()['counts'],{'PENDING':15})
    allids.extend(q.jobs);self.assertFalse(q.summary()['automatic_dispatch']);q.close()
   self.assertEqual(len(set(allids)),30)
 def test_missing_quality_cannot_reserve_gpu_work(self):
  with tempfile.TemporaryDirectory() as t:
   q=EchoLedger(Path(t)/'owner.sqlite','owner')
   with self.assertRaises(QueueBlocked):q.reserve(next(iter(q.jobs)),'example-nonce-0001')
   self.assertEqual(q.summary()['counts'],{'PENDING':15});q.close()
 def test_uncertain_execution_survives_restart_and_never_expires_into_retry(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'owner.sqlite';q=EchoLedger(p,'owner');job=next(iter(q.jobs))
   kwargs=dict(quality_confirmed=True,source_reverified=True,fresh_quota_seconds=36000,
       estimated_gpu_seconds=300,conditioning_png_sha256='a'*64)
   q.reserve(job,'example-nonce-0001',**kwargs);q.mark_submission(job,'example-nonce-0001',None);q.close()
   q=EchoLedger(p,'owner')
   self.assertEqual(q.summary()['counts'],{'PENDING':14,'UNCERTAIN':1})
   with self.assertRaises(QueueBlocked):q.reserve(job,'example-nonce-0002',**kwargs)
   backup=Path(t)/'backup.sqlite';self.assertEqual(len(q.snapshot(backup)),64);q.close()
 def test_account_identity_cannot_be_swapped_on_an_existing_queue(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'owner.sqlite';EchoLedger(p,'owner').close()
   with self.assertRaises(QueueBlocked):EchoLedger(p,'collaborator')
