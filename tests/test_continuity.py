"""Story identity and episode-order regression tests."""
import copy
import json
import unittest
from pathlib import Path
from studio.continuity import check_episode, check_sequence

ROOT=Path(__file__).resolve().parent.parent


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.e1=json.loads((ROOT/"episodes/E01_THE_REFLECTION.json").read_text(encoding="utf-8"))
        self.e2=json.loads((ROOT/"episodes/E02_THE_WRONG_WORLD.json").read_text(encoding="utf-8"))

    def test_real_pilots_are_consistent(self):
        a=check_sequence([self.e1,self.e2])
        self.assertEqual(len(a["episodes"]),2)
        self.assertEqual(a["episodes"][0]["exchange_count"],1)
        self.assertEqual(a["episodes"][1]["exchange_count"],0)

    def test_mira_cannot_be_in_earth_body_after_swap(self):
        copy_ep=copy.deepcopy(self.e1)
        copy_ep["scenes"][5]["continuity"]["occupant"]="Mira"
        with self.assertRaisesRegex(ValueError,"Lyra"):
            check_episode(copy_ep,strict=True)

    def test_no_secret_extra_exchange(self):
        copy_ep=copy.deepcopy(self.e2)
        copy_ep["scenes"][3]["continuity"]={"state":"transition","event":"mira_lyra_exchange"}
        with self.assertRaisesRegex(ValueError,"pre state"):
            check_episode(copy_ep,strict=True)

    def test_untracked_scenario_cannot_be_approved(self):
        with self.assertRaisesRegex(ValueError,"metadata"):
            check_episode({"id":"E88","scenes":[{"id":"S1"}]},strict=True)

    def test_episode_order_matters(self):
        with self.assertRaisesRegex(ValueError,"continuity break"):
            check_sequence([self.e2,self.e1])


if __name__ == "__main__":
    unittest.main()
