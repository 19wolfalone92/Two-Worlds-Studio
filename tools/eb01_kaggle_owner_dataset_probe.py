"""Read-only check whether the owner Kaggle account can see one exact private archive slug.

Never downloads media, creates/updates/deletes a dataset, submits GPU work, or
prints raw Kaggle CLI output. Only a small sanitized JSON receipt is emitted.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
from pathlib import Path

TARGET_DATASET = "wolf19loner92/eb01-owner-private-motion-a609d40bc3ee"


def parse_owner_dataset_csv(raw: str) -> dict:
    """Parse only exact ref presence; never return dataset titles or row data."""
    try:
        rows = list(csv.DictReader(io.StringIO(raw)))
    except (csv.Error, UnicodeError):
        return {"status": "OWNER_DATASET_LIST_OUTPUT_UNPARSEABLE",
                "exact_dataset_visible": None, "rows_examined": 0}
    refs = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ref = row.get("ref") or row.get("Ref") or row.get("datasetRef") or ""
        refs.append(str(ref).strip().lower())
    found = TARGET_DATASET.lower() in refs
    return {
        "status": ("OWNER_DATASET_LIST_OK_EXACT_SLUG_FOUND" if found
                   else "OWNER_DATASET_LIST_OK_EXACT_SLUG_NOT_FOUND"),
        "exact_dataset_visible": found,
        "rows_examined": len(refs),
    }


def classify_owner_list_failure(evidence: str) -> str:
    """Return a safe list-probe category; never expose raw CLI diagnostics."""
    text = str(evidence).lower()
    if re.search(r"\b403\b", text) or "forbidden" in text or "permission denied" in text:
        return "OWNER_DATASET_LIST_ACCESS_DENIED"
    if re.search(r"\b401\b", text) or "unauthorized" in text:
        return "OWNER_DATASET_LIST_AUTH_DENIED"
    if "no results found" in text or "no datasets found" in text:
        return "OWNER_DATASET_LIST_OK_EXACT_SLUG_NOT_FOUND"
    if "unrecognized arguments" in text or "no such option" in text:
        return "OWNER_DATASET_LIST_CLI_ARGUMENT_ERROR"
    from tools.eb01_archive_verified_private_pilot import classify_kaggle_failure
    return "OWNER_DATASET_LIST_" + classify_kaggle_failure(text)


def inspect_owner_dataset_visibility(runner=subprocess.run) -> dict:
    base = {
        "schema": 1,
        "target": TARGET_DATASET,
        "read_only": True,
        "dataset_created": False,
        "dataset_deleted": False,
        "media_downloaded": False,
        "gpu_submitted": False,
        "public_media_uploaded": False,
        "remote_acl_verified": False,
    }
    if not os.environ.get("KAGGLE_API_TOKEN"):
        return {**base, "status": "OWNER_KAGGLE_CREDENTIAL_MISSING"}
    try:
        help_result = runner(
            ["kaggle", "datasets", "list", "--help"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired:
        return {**base, "status": "OWNER_DATASET_LIST_HELP_TIMEOUT"}
    except OSError as exc:
        return {**base, "status": "OWNER_DATASET_LIST_CLIENT_" + type(exc).__name__}
    help_text = (help_result.stdout or "") + " " + (help_result.stderr or "")
    if help_result.returncode != 0:
        return {**base, "status": classify_owner_list_failure(help_text)}
    required = ("--mine", "--search", "--page", "--csv")
    missing = [flag for flag in required if flag not in help_text]
    if missing:
        return {**base, "status": "OWNER_DATASET_LIST_CLI_OPTIONS_MISSING",
                "missing_options": missing}
    try:
        result = runner(
            ["kaggle", "datasets", "list", "-m", "-s",
             "eb01-owner-private-motion-a609d40bc3ee", "-p", "1", "-v"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=75, check=False)
    except subprocess.TimeoutExpired:
        return {**base, "status": "OWNER_DATASET_LIST_TIMEOUT"}
    except OSError as exc:
        return {**base, "status": "OWNER_DATASET_LIST_CLIENT_" + type(exc).__name__}
    if result.returncode != 0:
        return {**base, "status": classify_owner_list_failure(
            (result.stdout or "") + " " + (result.stderr or ""))}
    parsed = parse_owner_dataset_csv(result.stdout or "")
    return {**base, **parsed}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_owner_dataset_visibility()
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if os.name == "posix":
        os.chmod(args.output, 0o600)
    print("EB01_KAGGLE_OWNER_DATASET_VISIBILITY", json.dumps(report, sort_keys=True))
    return 0 if report["status"] in (
        "OWNER_DATASET_LIST_OK_EXACT_SLUG_FOUND",
        "OWNER_DATASET_LIST_OK_EXACT_SLUG_NOT_FOUND") else 2


if __name__ == "__main__":
    raise SystemExit(main())
