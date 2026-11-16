"""Secure READ-ONLY dual-account Kaggle EB01 V8 status and scheduling audit.

Operates exclusively on separate account tokens supplied via GitHub Actions
protected secrets. Does not create datasets, share accounts, read files,
allocate GPUs, submit kernels, publish, or output credentials/usernames.
Do not treat this audit as owner approval to render or publish videos.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from studio.eb01_dual_kaggle_scheduler import (
    Worker, create_schedule, _least_load_assignment,
)
from studio.eb01_v8_40s_native_motion_plan import plan
from tools.eb01_kaggle2_auth_probe import (
    query, verify_brother_source_access,
)

VISIBLE = "ORIGINAL_OWNER_PRIVATE_DATASET_METADATA_VISIBLE"
AUTHENTICATED = "AUTHENTICATED_READ_ONLY"


def evaluate(primary_token: str, collaborator_token: str,
             *, runner=None) -> dict:
    kwargs = {"runner": runner} if runner is not None else {}
    one_state, one_handle = query(primary_token, **kwargs)
    two_state, two_handle = query(collaborator_token, **kwargs)
    distinct = (one_state == AUTHENTICATED and two_state == AUTHENTICATED
                and one_handle is not None and two_handle is not None
                and one_handle != two_handle)
    # Each token is used independently. Kaggle authorization to view a
    # shared dataset is a prerequisite; never copy/private-export sources.
    owner_acl = (verify_brother_source_access(primary_token, **kwargs)
                 if one_state == AUTHENTICATED
                 else "AUTH_NOT_VERIFIED")
    collaborator_acl = (verify_brother_source_access(collaborator_token, **kwargs)
                        if distinct else "DISTINCT_AUTH_NOT_VERIFIED")
    # No untrusted quota guess, no fabricated VRAM, no artistic self-approval.
    workers = {
        "owner": Worker(
            account_id="owner",
            authenticated=(one_state == AUTHENTICATED),
            private_source_acl_verified=(owner_acl == VISIBLE)),
        "collaborator": Worker(
            account_id="collaborator",
            authenticated=(two_state == AUTHENTICATED and distinct),
            private_source_acl_verified=(collaborator_acl == VISIBLE)),
    }
    status = create_schedule(workers)
    movie = plan()
    theoretical = _least_load_assignment(movie["scenes"],
                                          ["owner", "collaborator"])
    allocation = {
        "owner": sum(s["native_segment_count"] for s in movie["scenes"]
                     if theoretical[s["scene_id"]] == "owner"),
        "collaborator": sum(s["native_segment_count"] for s in movie["scenes"]
                            if theoretical[s["scene_id"]] == "collaborator"),
    }
    return {
        "schema": 1,
        "purpose": "EB01_V8_TWO_INDEPENDENT_COLLABORATOR_GPU_READ_ONLY_PREFLIGHT",
        "primary_account_auth": one_state,
        "second_account_auth": two_state,
        "two_distinct_authenticated_accounts": distinct,
        "primary_source_metadata": owner_acl,
        "second_source_metadata": collaborator_acl,
        "private_dataset_shared_with_second_account": collaborator_acl == VISIBLE,
        "hypothetical_balanced_scene_split_if_BOTH_are_ready": allocation,
        "account_blocking_gates": status["account_gates"],
        "next_gpu_take_proposals": status["planned_first_wave"],
        "measured_native_per_gpu_peak_vram_available": False,
        "measured_native_33frame_time_available": False,
        "confirmed_remaining_gpu_quota_available": False,
        "native_quality_pilot_approved": False,
        "t4x2_memory_pooled": False,
        "gpu_jobs_submitted": 0,
        "private_original_media_transferred": False,
        "public_media_uploaded": False,
        "movie_ready": False,
        "published": False,
        "state": ("DUAL_AUTH_AND_SHARED_SOURCE_METADATA_VISIBLE_NATIVE_PILOT_REQUIRED"
                  if distinct and owner_acl == VISIBLE and collaborator_acl == VISIBLE
                  else "BLOCKED_UNTIL_INDEPENDENT_ACCESS_AND_NATIVE_PILOT_VERIFIED"),
    }


def main() -> None:
    report = evaluate(
        os.environ.get("EB01_OWNER_KAGGLE_SECRET", ""),
        os.environ.get("EB01_COLLABORATOR_KAGGLE_SECRET", ""),
    )
    filename = os.environ.get("EB01_SAFE_STATUS_PATH")
    if not filename:
        raise RuntimeError("Missing private runner temporary output path")
    p = Path(filename)
    if not p.is_absolute():
        raise ValueError("Safe status must be stored outside the checkout")
    workdir = Path(os.environ.get("GITHUB_WORKSPACE", "/__none__")).resolve()
    if p.resolve().is_relative_to(workdir):
        raise ValueError("Refuse writing any account status inside public checkout")
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    if os.name == "posix":
        os.chmod(p, 0o600)
    # Output already sanitized; no Kaggle stdout, usernames or secret values.
    print("EB01_DUAL_KAGGLE_SAFE_STATUS",
          json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
