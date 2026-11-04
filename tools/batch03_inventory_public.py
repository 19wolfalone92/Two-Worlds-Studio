"""Read-only Kaggle production inventory with PUBLIC-SAFE aggregate output.

This never downloads private files, prints private file names,
kernel URLs, API responses, credentials, narration or image content.
The counts describe available ZIPs, not render-complete or approved films.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
from collections import defaultdict

OWNER = "wolf19loner92"
STORIES = {str(n) for n in range(13, 19)}
FILE_NAME = re.compile(r"TwoWorlds_PRIVATE_(1[3-8])_(image|voice|motion)\\.zip$", re.I)
APPROVED = re.compile(r"^wolf19loner92/(?:tw-batch03-[a-z0-9-]+|two-worlds-russian-(?:short|story)-gpu)$")
MAX_QUERIES = 60


def invoke(*args: str, timeout: int = 60) -> tuple[bool, str]:
    try:
        result = subprocess.run(["kaggle", *args], capture_output=True,
                                text=True, timeout=timeout, check=False)
        # Raw CLI output can carry private metadata and MUST NOT be logged.
        return result.returncode == 0, result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return False, ""


def list_owned_kernels() -> tuple[list[str], bool]:
    ok, raw = invoke("kernels", "list", "--mine", "--csv",
                     "--page-size", "100", "--sort-by", "dateRun")
    if not ok:
        return [], False
    try:
        lines = list(csv.DictReader(io.StringIO(raw)))
    except csv.Error:
        return [], False
    names = []
    for row in lines:
        handle = row.get("ref", row.get("Ref", ""))
        if isinstance(handle, str) and APPROVED.fullmatch(handle):
            names.append(handle)
    return list(dict.fromkeys(names))[:MAX_QUERIES], True


def discover_output_kinds(rows: list[dict]) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for row in rows:
        # Kaggle CLI release versions expose name, fileName or ref.
        for name in row.values():
            if not isinstance(name, str):
                continue
            match = FILE_NAME.search(name.strip().split("/")[-1])
            if match and match.group(1) in STORIES:
                found.add((match.group(1), match.group(2).lower()))
    return found


def inspect() -> dict:
    kernels, account_ok = list_owned_kernels()
    stage_stories: dict[str, set[str]] = defaultdict(set)
    inspected, failed = 0, 0
    for handle in kernels:
        # Never invoke a GPU-running or mutating command.
        ok, output = invoke("kernels", "files", handle, "--csv", "--page-size", "200")
        if not ok:
            failed += 1
            continue
        inspected += 1
        try:
            rows = list(csv.DictReader(io.StringIO(output)))
        except csv.Error:
            failed += 1
            continue
        for story, kind in discover_output_kinds(rows):
            stage_stories[kind].add(story)
    return {
        "schema": 1, "read_only": True, "gpu_requested": False,
        "paid_api_calls": False, "published": False,
        "account_query_succeeded": account_ok,
        "matching_kernels_examined": inspected,
        "kernel_file_queries_failed": failed,
        "image_zip_story_count": len(stage_stories["image"]),
        "voice_zip_story_count": len(stage_stories["voice"]),
        "motion_zip_story_count": len(stage_stories["motion"]),
        "image_and_voice_zip_story_count": len(stage_stories["image"] & stage_stories["voice"]),
        "total_required_stories": len(STORIES),
        "actual_rendered_mp4_verified": 0,
        "note": "ZIP existence alone is not proof of all scenes or valid published videos.",
    }


def main() -> int:
    # Fail closed if GitHub owner/credential are absent, never log secret values.
    if os.environ.get("KAGGLE_USERNAME") != OWNER or not os.environ.get("KAGGLE_API_TOKEN"):
        print(json.dumps({"read_only": True, "status": "MISSING_AUTH"}))
        return 2
    result = inspect()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["account_query_succeeded"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
