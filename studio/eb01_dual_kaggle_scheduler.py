"""Two independent Kaggle collaborators: deterministic, GPU-free EB01 V8 scheduler.

This module NEVER logs into Kaggle, allocates a GPU, downloads private media,
or approves a film. Every scheduled take is original I2V inference. Different
accounts execute disjoint, scene-sticky chains: no unfinished conditioning
frame ever crosses account boundaries. Capacity is independently measured per
GPU, not summed across T4 cards/accounts.

All inputs are *reviewed capability reports*, not claims of live availability.
An actual private runner MUST separately enforce authorization, source hashes,
verified predecessor PNG, quota, and original video validation at execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections.abc import Mapping
from typing import Any

SHA256 = re.compile(r"^[a-f0-9]{64}$")
OWNER = "owner"
COLLABORATOR = "collaborator"
ACCOUNT_IDS = (OWNER, COLLABORATOR)
# Keep a safety reserve for varying decode, model warmup and upload time.
QUOTA_FACTOR = 1.5
VRAM_RESERVE_GIB = 0.75
MAX_PARALLEL_JOBS_PER_ACCOUNT = 2


@dataclass(frozen=True)
class Worker:
    account_id: str
    authenticated: bool = False
    private_source_acl_verified: bool = False
    # Explicitly supplied by an independent per-account native-profile pilot.
    native_576x1024_pilot_approved: bool = False
    actual_gpu_seconds_per_33f: float | None = None
    measured_peak_gpu_vram_gib: float | None = None
    gpu_vram_per_device_gib: float | None = None
    gpu_device_count: int = 0
    available_quota_minutes: float | None = None
    # 2 independent processes are NEVER enabled merely by GPU T4x2 metadata.
    parallel_processes_tested: bool = False
    running_jobs: int = 0


def finite_positive(value: object) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and value > 0)


def worker_gates(worker: Worker) -> list[str]:
    errors = []
    if worker.account_id not in ACCOUNT_IDS:
        errors.append("INVALID_ACCOUNT_ALIAS")
    if worker.authenticated is not True:
        errors.append("ACCOUNT_AUTH_NOT_VERIFIED")
    if worker.private_source_acl_verified is not True:
        errors.append("PRIVATE_CANONICAL_SOURCE_ACCESS_NOT_VERIFIED")
    if worker.native_576x1024_pilot_approved is not True:
        errors.append("NATIVE_576X1024_HUMAN_REVIEWED_PILOT_MISSING")
    if not finite_positive(worker.actual_gpu_seconds_per_33f):
        errors.append("REAL_NATIVE_33F_GPU_SECONDS_NOT_MEASURED")
    if not finite_positive(worker.measured_peak_gpu_vram_gib):
        errors.append("PEAK_NATIVE_GPU_MEMORY_NOT_MEASURED")
    if not finite_positive(worker.gpu_vram_per_device_gib):
        errors.append("PER_DEVICE_GPU_VRAM_NOT_MEASURED")
    elif (finite_positive(worker.measured_peak_gpu_vram_gib) and
          worker.measured_peak_gpu_vram_gib + VRAM_RESERVE_GIB
          > worker.gpu_vram_per_device_gib):
        errors.append("NATIVE_PILOT_TOO_CLOSE_TO_PER_DEVICE_VRAM_LIMIT")
    if type(worker.gpu_device_count) is not int or not 1 <= worker.gpu_device_count <= 2:
        errors.append("GPU_DEVICE_COUNT_NOT_VERIFIED")
    if not finite_positive(worker.available_quota_minutes):
        errors.append("REMAINING_ACCOUNT_GPU_QUOTA_NOT_VERIFIED")
    if type(worker.running_jobs) is not int or worker.running_jobs < 0:
        errors.append("INVALID_ACTIVE_JOB_COUNT")
    return errors


def independent_slots(worker: Worker) -> int:
    """At most one active job per GPU; dual-GPU parallelism needs own benchmark."""
    if worker_gates(worker):
        return 0
    tested_slots = (min(MAX_PARALLEL_JOBS_PER_ACCOUNT, worker.gpu_device_count)
                    if worker.parallel_processes_tested is True else 1)
    return max(0, tested_slots - worker.running_jobs)


def _verified_tails(tails: Mapping[str, Any], scenes: list[dict]) -> dict[str, dict]:
    """Validate read-only cursor shape, NOT actual media/provenance.

    The private execution worker must independently authenticate these cursors
    against MP4 SHA, decoded frames, and original PNG bytes. A cursor alone can
    never authorize GPU work, release or artistic signoff.
    """
    scene_map = {s["scene_id"]: s for s in scenes}
    if not isinstance(tails, Mapping) or set(tails) - set(scene_map):
        raise ValueError("Unknown or malformed scene-cursor mapping")
    result = {}
    for scene_id, value in tails.items():
        if not isinstance(value, dict):
            raise ValueError("Scene cursor must be a dictionary")
        n = value.get("completed_count")
        last = value.get("next_conditioning_png_sha256")
        limit = scene_map[scene_id]["native_segment_count"]
        if type(n) is not int or n < 0 or n > limit:
            raise ValueError("Invalid completed native take count")
        if n and (not isinstance(last, str) or not SHA256.fullmatch(last)):
            raise ValueError("Missing exact last exported conditioning PNG digest")
        if n == 0 and last is not None:
            raise ValueError("Unverified initial source must not supply a tail hash")
        result[scene_id] = {"completed_count": n,
                            "next_conditioning_png_sha256": last}
    return result


def _least_load_assignment(scenes: list[dict], aliases: list[str]) -> dict[str, str]:
    """Whole-scene bin packing: no mid-scene migration or hidden PNG transfers."""
    if not aliases:
        return {}
    weights = {alias: 0 for alias in aliases}
    assignment = {}
    for scene in sorted(scenes, key=lambda s: (-s["native_segment_count"], s["scene_id"])):
        alias = min(aliases, key=lambda a: (weights[a], aliases.index(a)))
        assignment[scene["scene_id"]] = alias
        weights[alias] += scene["native_segment_count"]
    return assignment


def create_schedule(workers: Mapping[str, Worker],
                    verified_tails: Mapping[str, dict] | None = None,
                    in_flight_scenes: Mapping[str, list[str]] | None = None) -> dict:
    """Prepare non-authoritative next-take proposals for 1 or 2 accounts.

    NEVER submits a job. Do not feed this output directly into `kaggle push`.
    Its verified_tails input is merely a cursor for a trusted private worker;
    actual receipt/MP4 checks must run again at the point of use.
    """
    from studio.eb01_v8_40s_native_motion_plan import plan
    movie = plan()
    scenes = movie["scenes"]
    if not isinstance(workers, Mapping) or set(workers) != set(ACCOUNT_IDS):
        raise ValueError("Exactly owner and collaborator alias entries required")
    if any(not isinstance(workers[a], Worker) or
           workers[a].account_id != a for a in ACCOUNT_IDS):
        raise ValueError("Worker record does not match account alias")
    if in_flight_scenes is None:
        in_flight_scenes = {}
    if (not isinstance(in_flight_scenes, Mapping)
            or set(in_flight_scenes) - set(ACCOUNT_IDS)):
        raise ValueError("Invalid in-flight accounting")
    cursor = _verified_tails(verified_tails or {}, scenes)
    scene_ids = {s["scene_id"] for s in scenes}
    occupied = set()
    for account, ids in in_flight_scenes.items():
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError("Duplicate/non-list in-flight scene")
        for sid in ids:
            if sid not in scene_ids or sid in occupied:
                raise ValueError("In-flight scene duplicated across accounts")
            occupied.add(sid)
    blocked = {alias: worker_gates(workers[alias]) for alias in ACCOUNT_IDS}
    usable = [alias for alias in ACCOUNT_IDS if not blocked[alias]]
    assignment = _least_load_assignment(scenes, usable)
    planned_counts = {alias: 0 for alias in ACCOUNT_IDS}
    for scene in scenes:
        if scene["scene_id"] in assignment:
            planned_counts[assignment[scene["scene_id"]]] += scene["native_segment_count"]
    proposals = []
    for alias in usable:
        worker = workers[alias]
        slots = independent_slots(worker)
        # No per-account quota double counting: deduct for each candidate slot.
        projected_gpu_seconds = 0.0
        candidates = sorted(
            (scene for scene in scenes
             if assignment.get(scene["scene_id"]) == alias
             and scene["scene_id"] not in occupied
             and cursor.get(scene["scene_id"], {}).get("completed_count", 0)
                 < scene["native_segment_count"]),
            key=lambda s: (-(
                s["native_segment_count"] -
                cursor.get(s["scene_id"], {}).get("completed_count", 0)),
                s["scene_id"]))
        for scene in candidates:
            if slots <= 0:
                break
            sid = scene["scene_id"]
            n = cursor.get(sid, {}).get("completed_count", 0)
            source_sha = (
                scene["canonical_reference_sha256"] if n == 0
                else cursor[sid]["next_conditioning_png_sha256"])
            predicted_seconds = (worker.actual_gpu_seconds_per_33f
                                 * QUOTA_FACTOR)
            if projected_gpu_seconds + predicted_seconds > (
                    worker.available_quota_minutes * 60):
                break
            # Canonical hashes are committed values from the verified V8 plan.
            proposals.append({
                "account_alias": alias, "scene_id": sid,
                "native_take_index": n,
                "take_id": f"{sid}_TAKE{n:02d}",
                "expected_conditioning_png_sha256": source_sha,
                "canonical_reference_sha256": scene["canonical_reference_sha256"],
                "canonical_prompt_sha256": scene["canonical_prompt_sha256"],
                "canonical_voice_sha256": scene["canonical_voice_sha256"],
                "native_frames": 33, "fps": 24,
                "native_minimum_dimensions": [576, 1024],
                "estimated_reserved_gpu_seconds": round(predicted_seconds, 2),
                "requires_private_pre_submit_reverification": True,
                "gpu_submitted": False,
                "source_mp4_verified_by_scheduler": False,
            })
            projected_gpu_seconds += predicted_seconds
            slots -= 1
    return {
        "schema": 1,
        "episode": "ELIZABETH_BONES_E01_V8",
        "status": ("NEXT_TAKES_PLANNED_PRE_SUBMIT_RECHECK_REQUIRED"
                   if proposals else "NO_GPU_JOB_AUTHORIZED"),
        "account_gates": blocked,
        "scene_assignments": assignment,
        "whole_scene_take_split": planned_counts,
        "native_original_take_count": movie["new_true_inference_segment_count"],
        "planned_first_wave": proposals,
        "gpu_jobs_submitted": 0,
        "private_media_exported": False,
        "two_t4_devices_have_pooled_vram": False,
        "media_verified": False,
        "full_movie_ready": False,
        "published": False,
    }
