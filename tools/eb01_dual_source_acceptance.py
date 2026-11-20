"""Read-only, independently authenticated byte-level acceptance of shared EB01 inputs.

No dataset creation, no sharing mutation, no GPU and no public media artifacts.
Only numerical receipts leave each private temporary directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from tools.eb01_kaggle2_auth_probe import query, SOURCE_DATASET
from tools.eb01_private_two_account_dataset_sync import (
    FILE_SHA, SourceSyncBlocked, kaggle, validated_extract,
)


from tools.eb01_kaggle_token_env import isolated_environment


def verify_account(token: str, root: Path, *, api=kaggle) -> dict:
    result = {"listing_verified": False, "download_verified": False,
              "canonical_sha_verified": 0, "http_403_seen": False}
    try:
        api(token, ["kaggle", "datasets", "files", SOURCE_DATASET], timeout=90)
        result["listing_verified"] = True
        dest = root / "download"
        dest.mkdir(mode=0o700, parents=True)
        api(token, ["kaggle", "datasets", "download", "-d", SOURCE_DATASET,
                    "-p", str(dest), "--force", "--quiet"], timeout=360)
        archives = list(dest.glob("*.zip"))
        if len(archives) != 1:
            raise SourceSyncBlocked("SOURCE_ARCHIVE_MISSING_OR_AMBIGUOUS")
        result["download_verified"] = True
        checks = validated_extract(archives[0], root / "verified")
        result["canonical_sha_verified"] = checks["sources_sha_verified"]
        result["status"] = "SOURCE_DOWNLOAD_AND_ALL_CANONICAL_SHA_VERIFIED"
        result["canonical_manifest_sha256"] = hashlib.sha256(
            json.dumps(FILE_SHA, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    except SourceSyncBlocked as exc:
        result["http_403_seen"] = exc.code == "ACCESS_DENIED"
        result["status"] = "BLOCKED_" + exc.code
    finally:
        # Do not retain portraits/audio even when an account or hash check fails.
        shutil.rmtree(root, ignore_errors=True)
    return result


def accept(owner_token: str, second_token: str, root: Path, *, auth=query, api=kaggle) -> dict:
    report = {"schema": 1, "two_distinct_authenticated_accounts": False,
              "all_eight_sources_verified_for_both": False,
              "gpu_jobs_submitted": 0, "dataset_mutations": 0,
              "public_media_uploaded": False, "movie_ready": False,
              "source_dataset": SOURCE_DATASET}
    if not owner_token or not second_token or owner_token == second_token:
        report["status"] = "BLOCKED_TWO_DISTINCT_TOKENS_REQUIRED"
        return report
    a, owner = auth(owner_token)
    b, second = auth(second_token)
    report["authentication"] = {"owner": a, "collaborator": b}
    distinct = (a == b == "AUTHENTICATED_READ_ONLY" and bool(owner) and bool(second)
                and owner.lower() != second.lower())
    report["two_distinct_authenticated_accounts"] = distinct
    if not distinct:
        report["status"] = "BLOCKED_DISTINCT_AUTHENTICATION_REQUIRED"
        return report
    report["accounts"] = {
        "owner": verify_account(owner_token, root / "owner", api=api),
        "collaborator": verify_account(second_token, root / "collaborator", api=api),
    }
    report["all_eight_sources_verified_for_both"] = all(
        x.get("canonical_sha_verified") == len(FILE_SHA) and
        x.get("status") == "SOURCE_DOWNLOAD_AND_ALL_CANONICAL_SHA_VERIFIED"
        for x in report["accounts"].values())
    report["status"] = ("BOTH_ACCOUNTS_SHARED_SOURCE_BYTES_SHA_VERIFIED"
                        if report["all_eight_sources_verified_for_both"]
                        else "BLOCKED_SHARED_SOURCE_ACCEPTANCE")
    return report


def main() -> int:
    os.umask(0o077)
    root = Path(os.environ["RUNNER_TEMP"]).resolve()
    checkout = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    if root == checkout or root.is_relative_to(checkout):
        raise RuntimeError("Private temporary root must be outside public checkout")
    with tempfile.TemporaryDirectory(dir=root, prefix="eb01-source-acceptance-") as temp:
        report = accept(os.environ.get("EB01_OWNER_TOKEN", ""),
                        os.environ.get("EB01_SECOND_TOKEN", ""), Path(temp))
    path = root / "EB01_DUAL_SOURCE_ACCEPTANCE_SAFE.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("EB01_DUAL_SOURCE_ACCEPTANCE", json.dumps(report, sort_keys=True))
    return 0 if report["all_eight_sources_verified_for_both"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
