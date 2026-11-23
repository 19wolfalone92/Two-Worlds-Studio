"""Read ONLY existing pinned private Kaggle anatomy MP4s; no new GPU jobs.

Original Kaggle notebook identity is derived from the exact successful
GitHub Actions run. Raw media never leave the ephemeral private runner;
only a bounded, path-free numerical summary is returned to GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from studio.eb01_anatomy_profile import anatomy_specs
from studio.eb01_regional_motion import compare_paired_motion
from tools.eb01_echo_parallel_pilot import verify_slot
from tools.eb01_kaggle2_auth_probe import query
from tools.eb01_native_two_account_pilot import call, STATE, PilotBlocked

ORIGINAL_RUN_ID = "45112430157"
ORIGINAL_REVISION = "5444d5cd9a5beaadb92b5dc4145f028b4b9cb08c"


def original_kernel(alias: str, handle: str) -> str:
    if alias not in ("owner", "collaborator"):
        raise ValueError("Unknown account")
    # Exact immutable slug used by the successful previous benchmark.
    nonce = hashlib.sha256(
        ("eb01-anatomy-v1-" + ORIGINAL_RUN_ID + "-" + alias).encode()
    ).hexdigest()[:14]
    return handle + "/eb01-anatomy-" + nonce


def compare_existing(alias: str, token: str) -> dict:
    safe = {
        "schema": 1,
        "account_alias": alias,
        "original_run_id": ORIGINAL_RUN_ID,
        "original_revision": ORIGINAL_REVISION,
        "new_kaggle_gpu_submissions": 0,
        "full_production_authorized": False,
        "human_artistic_approved": False,
    }
    if alias not in ("owner", "collaborator"):
        raise ValueError("Invalid account")
    status, handle = query(token)
    if status != "AUTHENTICATED_READ_ONLY" or not handle:
        raise PilotBlocked("EXISTING_PILOT_ACCOUNT_AUTH_UNAVAILABLE")
    kernel = original_kernel(alias, handle)
    states = STATE.findall(call(token, ["kaggle", "kernels", "status", kernel], timeout=65))
    if states != ["COMPLETE"]:
        raise PilotBlocked("EXISTING_PILOT_NOT_COMPLETE_OR_UNAVAILABLE")
    # Separate temporary directory for each account; never archive or upload raw media.
    with tempfile.TemporaryDirectory(dir=os.environ.get("RUNNER_TEMP"),
                                     prefix="eb01-previous-native-readonly-") as temp:
        output = Path(temp) / "raw"
        output.mkdir(mode=0o700)
        call(token, ["kaggle", "kernels", "output", kernel,
                     "-p", str(output), "--force"], timeout=480)
        variants = []
        for slot, spec in enumerate(anatomy_specs(alias)):
            proof = verify_slot(output / ("slot" + str(slot)), spec, ORIGINAL_REVISION)
            if not proof.get("technical_pass") or not isinstance(proof.get("regional_motion"), dict):
                raise PilotBlocked("ORIGINAL_MP4_OR_REGION_QA_UNVERIFIED")
            variants.append(proof["regional_motion"])
        summary = compare_paired_motion(variants[0], variants[1])
        safe.update(
            status="EXISTING_REAL_MP4S_REMEASURED",
            shot=summary["shot"],
            original_mp4_count=2,
            decoded_frames=[v["decoded_frames"] for v in variants],
            visual_flags=[v["review_flags"] for v in variants],
            paired_anatomy_metric=summary,
            full_production_authorized=False,
            human_artistic_approved=False,
        )
    return safe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", choices=("owner", "collaborator"), required=True)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        report = compare_existing(args.account, os.environ.get("EB01_ACCOUNT_TOKEN", ""))
        code = 0
    except Exception as exc:  # No private subprocess/path/credential traceback in public Actions.
        # No raw CLI error text, file paths, usernames, tokens, or confidential data.
        report = {
            "schema": 1, "account_alias": args.account,
            "original_run_id": ORIGINAL_RUN_ID,
            "status": "EXISTING_REAL_PILOT_COMPARISON_BLOCKED",
            "error_type": type(exc).__name__,
            "new_kaggle_gpu_submissions": 0,
            "full_production_authorized": False,
            "human_artistic_approved": False,
        }
        code = 2
    out = Path(os.environ["RUNNER_TEMP"]) / (
        "EB01_EXISTING_ANATOMY_" + args.account + "_SAFE.json"
    )
    out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("EB01_EXISTING_ANATOMY_COMPARISON", json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
