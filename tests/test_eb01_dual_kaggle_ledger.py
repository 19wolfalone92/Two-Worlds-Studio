import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from studio.eb01_dual_kaggle_ledger import Ledger,LedgerBlocked,production_manifest
from studio.eb01_dual_kaggle_scheduler import Worker


def ready(a):
    return Worker(a,authenticated=True,private_source_acl_verified=True,
        native_576x1024_pilot_approved=True,actual_gpu_seconds_per_33f=337,
        measured_peak_gpu_vram_gib=12.69,gpu_vram_per_device_gib=14.562,
        gpu_device_count=2,available_quota_minutes=100)

class DurableSceneQueueTests(unittest.TestCase):
    def test_scene_sticky_exact_15_15_and_distinct_private_queues(self):
        doc=production_manifest();self.assertEqual(doc['takes_per_account'],{'owner':15,'collaborator':15})
        self.assertFalse(doc['automatic_gpu_dispatch']);sets=[]
        with tempfile.TemporaryDirectory() as t:
            for alias in ('owner','collaborator'):
                a=Ledger(Path(t)/(alias+'.sqlite'),alias)
                sets.append({r[0] for r in a.connection.execute('SELECT job_id FROM jobs')})
                self.assertEqual(a.summary()['counts'],{'PENDING':15});a.close()
        self.assertFalse(sets[0]&sets[1]);self.assertEqual(len(sets[0]|sets[1]),30)

    def test_atomic_claim_reopen_and_uncertain_submission_never_duplicate(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'owner.sqlite';a=Ledger(path,'owner');b=Ledger(path,'owner')
            job=a.connection.execute('SELECT job_id FROM jobs WHERE segment_index=0 LIMIT 1').fetchone()[0]
            a.reserve(job,'immutable-job-001',ready('owner'))
            with self.assertRaises(LedgerBlocked):b.reserve(job,'duplicate-job-002',ready('owner'))
            a.mark_submission(job,'immutable-job-001',None);a.close();b.close()
            c=Ledger(path,'owner');self.assertEqual(c.summary()['counts']['UNCERTAIN'],1)
            with self.assertRaises(LedgerBlocked):c.reserve(job,'retry-job-003',ready('owner'))
            c.close()

    def test_unknown_quality_wrong_account_and_insufficient_quota_block(self):
        with tempfile.TemporaryDirectory() as t:
            a=Ledger(Path(t)/'owner.sqlite','owner')
            job=a.connection.execute('SELECT job_id FROM jobs WHERE segment_index=0 LIMIT 1').fetchone()[0]
            for worker in (Worker('owner'),ready('collaborator'),replace(ready('owner'),available_quota_minutes=1),replace(ready('owner'),running_jobs=1)):
                with self.assertRaises(LedgerBlocked):a.reserve(job,'blocked-job-001',worker)
            self.assertEqual(a.summary()['counts'],{'PENDING':15});a.close()

    def test_predecessor_cannot_be_skipped(self):
        with tempfile.TemporaryDirectory() as t:
            a=Ledger(Path(t)/'owner.sqlite','owner')
            job=a.connection.execute('SELECT job_id FROM jobs WHERE segment_index=1 LIMIT 1').fetchone()[0]
            with self.assertRaises(LedgerBlocked):a.reserve(job,'future-job-001',ready('owner'),predecessor_png_sha256='a'*64)
            a.close()

    def test_database_account_identity_is_immutable(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'owner.sqlite';a=Ledger(path,'owner');a.close()
            with self.assertRaises(LedgerBlocked):Ledger(path,'collaborator')

if __name__=='__main__':unittest.main()
