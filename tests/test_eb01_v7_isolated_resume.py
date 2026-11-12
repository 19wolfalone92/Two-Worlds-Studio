"""CPU-only regression for resuming only missing EB01 shots.

The original 7-shot Kaggle 3600s job was canceled after two real takes:
S00 (49 frames) and S01 (61). Never repeat those originals or declare
a 2-shot partial render to be the final film.
"""
import unittest
from tools.kaggle_eb01_v7_single_shot import ALLOWED,check_configuration
from studio.eb01_v7_storyboard import SHOTS,SRC_SHA,VOICE_SHA
from tools.kaggle_eb01_v7_full_entry import REVIEWED_SOURCE_SHA

BASE={
    "schema":7,"shot":"S02","model":"wan22-5b","private":True,
    "new_gpu_jobs":1,"source_dataset":"wolf19loner92/eb01-v7-private-production-sources",
    "allow_social_publish":False,"allow_regenerate_good_shots":False,
    "revision":"a"*40,"branch":"fix/eb01-v7-resume-existing-clips-20261112",
    "nonce":"eb01-v7-s02-local-smoke"
}

class PerShotResumeTests(unittest.TestCase):
    def test_generates_only_missing_five_original_shots(self):
        self.assertEqual(ALLOWED,{"S02","S03","S04","S05","S06"})
        expected={s[0] for s in SHOTS if s[0] not in ("S00","S01")}
        self.assertEqual(ALLOWED,expected)
        self.assertEqual(len(SRC_SHA),2)
        self.assertEqual(len(VOICE_SHA),6)
        self.assertEqual(set(REVIEWED_SOURCE_SHA),{s[0] for s in SHOTS})
        for sid in sorted(ALLOWED):
            c=dict(BASE,shot=sid)
            self.assertEqual(check_configuration(c),sid)
    def test_good_existing_scenes_are_never_regenerated(self):
        for sid in ("S00","S01","S07",""):
            with self.subTest(sid=sid):
                with self.assertRaises(ValueError):
                    check_configuration(dict(BASE,shot=sid))
    def test_public_model_or_multiple_jobs_are_blocked(self):
        for patch in [
            {"private":False},{"new_gpu_jobs":2},{"allow_social_publish":True},
            {"allow_regenerate_good_shots":True},{"model":"fixture"},
            {"revision":"b"*39},
            {"source_dataset":"public/raw-video"}
        ]:
            with self.subTest(patch=patch):
                with self.assertRaises(ValueError):
                    check_configuration({**BASE,**patch})
    def test_single_shot_model_budget(self):
        shots={s[0]:s for s in SHOTS}
        # Strict four native guard frames and 3600s Kaggle limit.
        for sid in sorted(ALLOWED):
            frames=shots[sid][2]
            self.assertEqual((frames-1)%4,0)
            self.assertLessEqual(frames+4,133)
            self.assertGreaterEqual(frames/24,shots[sid][3])
            self.assertGreater(len(shots[sid][-1]),150)

if __name__=="__main__":unittest.main()
