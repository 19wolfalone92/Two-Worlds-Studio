"""No-GPU regression for independent Kaggle owner+collaborator EB01 V8 scheduling."""
import copy
import unittest

from studio.eb01_dual_kaggle_scheduler import (
    Worker, create_schedule, independent_slots, worker_gates,
    _least_load_assignment,
)
from studio.eb01_v8_40s_native_motion_plan import plan


def capable(account, **changes):
    fields = dict(
        account_id=account, authenticated=True,
        private_source_acl_verified=True,
        native_576x1024_pilot_approved=True,
        actual_gpu_seconds_per_33f=500.0,
        measured_peak_gpu_vram_gib=12.8,
        gpu_vram_per_device_gib=14.56,
        gpu_device_count=2,
        available_quota_minutes=180,
        parallel_processes_tested=False,
        running_jobs=0)
    fields.update(changes)
    return Worker(**fields)


class SceneAssignmentTests(unittest.TestCase):
    def test_30_original_takes_balance_exactly_between_two_verified_accounts(self):
        workers = {"owner": capable("owner"),
                   "collaborator": capable("collaborator")}
        result = create_schedule(workers)
        self.assertEqual(result["native_original_take_count"], 30)
        self.assertEqual(result["whole_scene_take_split"],
                         {"owner": 15, "collaborator": 15})
        self.assertEqual(len(result["scene_assignments"]), 7)
        self.assertEqual(len(result["planned_first_wave"]), 2)
        self.assertEqual(set(x["account_alias"] for x in result["planned_first_wave"]),
                         {"owner", "collaborator"})
        self.assertEqual(result["gpu_jobs_submitted"], 0)
        self.assertFalse(result["media_verified"])
        self.assertFalse(result["full_movie_ready"])
        self.assertFalse(result["two_t4_devices_have_pooled_vram"])

    def test_second_account_without_private_dataset_acl_gets_zero_source_shots(self):
        workers = {"owner": capable("owner"),
                   "collaborator": capable("collaborator",
                                            private_source_acl_verified=False)}
        result = create_schedule(workers)
        self.assertEqual(result["whole_scene_take_split"],
                         {"owner": 30, "collaborator": 0})
        self.assertEqual(len(result["planned_first_wave"]), 1)
        self.assertIn("PRIVATE_CANONICAL_SOURCE_ACCESS_NOT_VERIFIED",
                      result["account_gates"]["collaborator"])

    def test_current_low_resolution_pilot_does_not_unlock_new_native_batch(self):
        workers = {"owner": capable("owner", native_576x1024_pilot_approved=False),
                   "collaborator": capable("collaborator",
                                            private_source_acl_verified=False)}
        result = create_schedule(workers)
        self.assertEqual(result["status"], "NO_GPU_JOB_AUTHORIZED")
        self.assertEqual(result["planned_first_wave"], [])
        self.assertEqual(result["scene_assignments"], {})

    def test_first_takes_bind_exact_canonical_actor_voice_and_prompt(self):
        workers = {"owner": capable("owner"),
                   "collaborator": capable("collaborator")}
        result = create_schedule(workers)
        by_scene = {s["scene_id"]: s for s in plan()["scenes"]}
        for task in result["planned_first_wave"]:
            scene = by_scene[task["scene_id"]]
            self.assertEqual(task["native_take_index"], 0)
            self.assertEqual(task["expected_conditioning_png_sha256"],
                             scene["canonical_reference_sha256"])
            self.assertEqual(task["canonical_voice_sha256"],
                             scene["canonical_voice_sha256"])
            self.assertEqual(task["canonical_prompt_sha256"],
                             scene["canonical_prompt_sha256"])
            self.assertFalse(task["source_mp4_verified_by_scheduler"])
            self.assertFalse(task["gpu_submitted"])

    def test_resume_take_one_requires_verified_png_tail_and_scene_is_sticky(self):
        workers = {"owner": capable("owner"),
                   "collaborator": capable("collaborator")}
        initial = create_schedule(workers)
        task = initial["planned_first_wave"][0]
        scene = task["scene_id"]
        alias = task["account_alias"]
        resumed = create_schedule(
            workers,
            verified_tails={scene: {"completed_count": 1,
                                    "next_conditioning_png_sha256": "a"*64}},
            scene_owner_locks={scene: alias})
        self.assertEqual(resumed["scene_assignments"][scene], alias)
        relevant = [x for x in resumed["planned_first_wave"]
                    if x["account_alias"] == alias]
        self.assertTrue(relevant)
        self.assertEqual(relevant[0]["scene_id"], scene)
        self.assertEqual(relevant[0]["native_take_index"], 1)
        self.assertEqual(relevant[0]["expected_conditioning_png_sha256"], "a"*64)

    def test_no_same_scene_simultaneously_across_accounts(self):
        workers = {"owner": capable("owner"),
                   "collaborator": capable("collaborator")}
        initial = create_schedule(workers)
        busy = initial["planned_first_wave"][0]
        after = create_schedule(workers,
            in_flight_scenes={busy["account_alias"]: [busy["scene_id"]]},
            scene_owner_locks={busy["scene_id"]:busy["account_alias"]})
        self.assertNotIn(busy["scene_id"],
                         [t["scene_id"] for t in after["planned_first_wave"]])

    def test_parallel_on_two_t4_requires_separate_real_benchmark(self):
        normal = capable("owner")
        self.assertEqual(independent_slots(normal), 1)
        ready = capable("owner", parallel_processes_tested=True)
        self.assertEqual(independent_slots(ready), 2)
        workers = {"owner": ready,
                   "collaborator": capable("collaborator",
                                             parallel_processes_tested=True)}
        result = create_schedule(workers)
        self.assertEqual(len(result["planned_first_wave"]), 4)
        self.assertEqual(len({x["scene_id"] for x in result["planned_first_wave"]}), 4)

    def test_active_jobs_and_reserved_quota_prevent_oversubmission(self):
        running = capable("owner", parallel_processes_tested=True,
                          running_jobs=2)
        self.assertEqual(independent_slots(running), 0)
        workers = {"owner": running,
                   "collaborator": capable("collaborator",
                                            available_quota_minutes=8)}
        result = create_schedule(workers)
        self.assertEqual(len(result["planned_first_wave"]), 0)
        self.assertEqual(result["gpu_jobs_submitted"], 0)
        # 500*1.5=750 seconds reserved; 8 minutes is insufficient.

    def test_quota_guard_and_per_card_vram_not_summed(self):
        for wrong in (0, True, -2, float("nan"), float("inf")):
            self.assertIn("REMAINING_ACCOUNT_GPU_QUOTA_NOT_VERIFIED",
                          worker_gates(capable("owner", available_quota_minutes=wrong)))
        bad = capable("owner", measured_peak_gpu_vram_gib=17.0,
                      gpu_vram_per_device_gib=14.56, gpu_device_count=2)
        self.assertIn("NATIVE_PILOT_TOO_CLOSE_TO_PER_DEVICE_VRAM_LIMIT",
                      worker_gates(bad))
        self.assertEqual(independent_slots(bad), 0)

    def test_bad_predecessor_receipts_do_not_propose_continuation(self):
        workers = {"owner": capable("owner"), "collaborator": capable("collaborator")}
        for tail in ("g"*64, True, None):
            with self.assertRaises(ValueError):
                create_schedule(workers,
                    verified_tails={"S02": {"completed_count": 1,
                                            "next_conditioning_png_sha256": tail}})
        with self.assertRaises(ValueError):
            create_schedule(workers,
                verified_tails={"S02": {"completed_count": 40,
                                        "next_conditioning_png_sha256": "a"*64}})

    def test_running_scene_never_migrates_to_other_account_on_acl_or_quota_loss(self):
        workers = {"owner": capable("owner"),
                   "collaborator": capable("collaborator")}
        initial = create_schedule(workers)
        assigned_to_second = next(s for s, owner in initial["scene_assignments"].items()
                                  if owner == "collaborator")
        now_offline = {"owner": capable("owner"),
                       "collaborator": capable("collaborator",
                                                private_source_acl_verified=False)}
        paused = create_schedule(now_offline,
            verified_tails={assigned_to_second:
                {"completed_count": 1, "next_conditioning_png_sha256": "d"*64}},
            scene_owner_locks={assigned_to_second: "collaborator"})
        self.assertEqual(paused["scene_assignments"][assigned_to_second],
                         "collaborator")
        self.assertNotIn(assigned_to_second,
            [job["scene_id"] for job in paused["planned_first_wave"]])
        self.assertIn("PRIVATE_CANONICAL_SOURCE_ACCESS_NOT_VERIFIED",
                      paused["account_gates"]["collaborator"])

    def test_started_scene_without_persisted_owner_lock_is_refused(self):
        workers = {"owner": capable("owner"),
                   "collaborator": capable("collaborator")}
        with self.assertRaises(ValueError):
            create_schedule(workers,
                verified_tails={"S02": {"completed_count": 1,
                                        "next_conditioning_png_sha256": "a"*64}})
        with self.assertRaises(ValueError):
            create_schedule(workers,
                in_flight_scenes={"owner": ["S02"]},
                scene_owner_locks={"S02": "collaborator"})

    def test_wrong_alias_or_duplicate_inflight_is_rejected(self):
        workers = {"owner": capable("owner"),
                   "collaborator": capable("collaborator")}
        with self.assertRaises(ValueError):
            create_schedule({"owner": workers["owner"]})
        with self.assertRaises(ValueError):
            create_schedule(workers,
                in_flight_scenes={"owner": ["S02"], "collaborator": ["S02"]})
        with self.assertRaises(ValueError):
            create_schedule(workers, in_flight_scenes={"owner": ["S02", "S02"]})


if __name__ == "__main__":
    unittest.main()
