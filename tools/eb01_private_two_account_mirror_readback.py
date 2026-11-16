"""Read-only cross-account Kaggle dataset custody proof, using separate tokens.

Intended after an owner-authorized CREATE-ONLY private mirror. Never uploads,
updates, publishes or allocates GPU. Original Kaggle files remain on each
account. Public GitHub Actions sees only sanitized no-media numerical status.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from tools.eb01_kaggle2_auth_probe import query
from tools.eb01_private_two_account_dataset_sync import (
    MIRROR_PREFIX, SourceSyncBlocked, FILE_SHA, kaggle, validated_extract,
)


def mirror_id(handle: str, create_run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,29}", handle):
        raise SourceSyncBlocked("INVALID_SECOND_ACCOUNT_HANDLE")
    if not re.fullmatch(r"[1-9][0-9]{5,20}", create_run_id):
        raise SourceSyncBlocked("UNPINNED_CREATE_RUN_ID")
    suffix = hashlib.sha256(("eb01-mirror-" + create_run_id).encode()).hexdigest()[:12]
    return handle + "/" + MIRROR_PREFIX + "-" + suffix


def verify_readback(first_token: str, second_token: str,
                    create_run_id: str, root: Path) -> dict:
    report = {
        "schema": 1, "run_id_locked": True,
        "independent_accounts": False,
        "second_source_files_sha_verified": 0,
        "private_acl_negative_control_pass": False,
        "source_files_exposed_to_public_github": False,
        "source_unchanged": True, "gpu_submitted": False,
        "published": False, "full_movie_ready": False,
    }
    try:
        auth1, handle1 = query(first_token)
        auth2, handle2 = query(second_token)
        if (auth1 != "AUTHENTICATED_READ_ONLY" or
            auth2 != "AUTHENTICATED_READ_ONLY" or
            not handle1 or not handle2 or handle1 == handle2):
            raise SourceSyncBlocked("DISTINCT_KAGGLE_AUTH_MISSING")
        report["independent_accounts"] = True
        ref = mirror_id(handle2, create_run_id)
        # The rightful second owner MUST have explicit read capability.
        kaggle(second_token, ["kaggle", "datasets", "files", ref],
               timeout=120)
        # Independent negative-control: the FIRST account should NOT see
        # the SECOND account's new private dataset, even though it owns
        # the source originals. If it can, privacy is not proven.
        try:
            kaggle(first_token, ["kaggle", "datasets", "files", ref],
                   timeout=90)
        except SourceSyncBlocked as exc:
            if exc.code not in ("NOT_FOUND", "ACCESS_DENIED"):
                raise SourceSyncBlocked("NEGATIVE_ACL_" + exc.code) from None
        else:
            raise SourceSyncBlocked("MIRROR_READABLE_WITH_UNINVITED_FIRST_ACCOUNT")
        report["private_acl_negative_control_pass"] = True
        download = root / "readback"
        download.mkdir(mode=0o700, parents=True, exist_ok=True)
        kaggle(second_token,
               ["kaggle", "datasets", "download", "-d", ref,
                "-p", str(download), "--force", "--quiet"],
               timeout=420)
        archives = list(download.glob("*.zip"))
        if len(archives) != 1:
            raise SourceSyncBlocked("SECOND_ACCOUNT_MIRROR_ZIP_NOT_FOUND")
        measured = validated_extract(archives[0], root / "verified",
                                     FILE_SHA)
        if measured["sources_sha_verified"] != 8:
            raise SourceSyncBlocked("NOT_EIGHT_SHA_VERIFIED_SOURCE_FILES")
        report["second_source_files_sha_verified"] = measured["sources_sha_verified"]
        report["status"] = "SECOND_PRIVATE_ACL_AND_ALL_EIGHT_SOURCE_SHAS_VERIFIED"
    except SourceSyncBlocked as exc:
        report["status"] = "BLOCKED_" + exc.code
    return report


def main() -> int:
    os.umask(0o077)
    tmp = os.environ.get("RUNNER_TEMP")
    if not tmp:
        raise RuntimeError("Readback requires private GitHub temporary runner")
    root = Path(tmp).resolve()
    workspace = Path(os.environ.get("GITHUB_WORKSPACE", "/__no__")).resolve()
    if workspace == root or workspace in root.parents or root in workspace.parents:
        raise RuntimeError("Temporary media cannot be checked out to public GitHub")
    with tempfile.TemporaryDirectory(prefix="eb01-mirror-readback-",dir=root) as d:
        report = verify_readback(
            os.environ.get("EB01_OWNER_TOKEN", ""),
            os.environ.get("EB01_SECOND_TOKEN", ""),
            os.environ.get("EB01_MIRROR_CREATE_RUN_ID", ""),
            Path(d))
    dest = root / "EB01_MIRROR_CROSS_ACCOUNT_READBACK_SAFE.json"
    dest.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
    dest.chmod(0o600)
    print("EB01_SECOND_ACCOUNT_MIRROR_SAFE_READBACK",
          json.dumps(report, sort_keys=True))
    return 0 if report["status"] == (
        "SECOND_PRIVATE_ACL_AND_ALL_EIGHT_SOURCE_SHAS_VERIFIED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
