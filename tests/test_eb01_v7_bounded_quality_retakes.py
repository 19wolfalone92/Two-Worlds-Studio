"""CPU-only regressions for bounded, identity-locked genuine Wan retakes."""
import unittest
from tools.kaggle_eb01_v7_single_shot import check_configuration
from tools.eb01_v7_auto_finish import BRANCH, SOURCE_DATASET

BASE={
    "schema":7,"shot":"S05","model":"wan22-5b","private":True,
    "new_gpu_jobs":1,"source_dataset":SOURCE_DATASET,
    "allow_social_publish":False,"allow_regenerate_good_shots":False,
    "revision":"b"*40,"branch":BRANCH,
    "nonce":"unit-test-retry"
}

class GenuineWanRetakeConfigTests(unittest.TestCase):
    def test_canonical_resume_valid(self):
        for sid in ("S02","S03","S04","S05","S06"):
            with self.subTest(sid=sid):
                self.assertEqual(check_configuration(dict(BASE,shot=sid)),sid)

    def test_rejected_s05_s02_and_failed_s06_can_retake_with_new_seed(self):
        for sid in ("S05","S02","S06"):
            for variant in (1,2):
                with self.subTest(sid=sid,variant=variant):
                    cfg=dict(BASE,shot=sid,quality_retake=True,seed_variant=variant)
                    self.assertEqual(check_configuration(cfg),sid)

    def test_no_unbounded_automatic_gpu_or_nonretake_variant(self):
        for patch in (
            {"quality_retake":True,"seed_variant":3},
            {"quality_retake":True,"seed_variant":-1},
            {"quality_retake":True,"seed_variant":False},
            {"quality_retake":True,"seed_variant":"2"},
            {"quality_retake":False,"seed_variant":1},
            {"quality_retake":1,"seed_variant":1},
            {"shot":"S03","quality_retake":True,"seed_variant":1},
            {"shot":"S04","quality_retake":True,"seed_variant":1},
            {"shot":"S01","quality_retake":True,"seed_variant":1},
            {"allow_social_publish":True},
            {"private":False},
            {"new_gpu_jobs":4},
        ):
            with self.subTest(patch=patch):
                with self.assertRaises(ValueError):
                    check_configuration({**BASE,**patch})

if __name__=="__main__":
    unittest.main()
