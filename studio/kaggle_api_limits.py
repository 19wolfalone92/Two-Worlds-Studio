"""Kaggle API budget facts and safe *one-attempt* CLI access.

This is NOT a synthetic 'N requests/minute' meter: Kaggle publishes a dynamic
rate limiter with no fixed numeric API-request allowance. GPU/TPU hours, shown
by `kaggle quota`, are a DIFFERENT resource. Avoid blindly retrying 429.
All commands are a fixed allowlist and stderr is never echoed to public CI.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

KERNEL = re.compile(r"^[A-Za-z0-9_-]{2,30}/[a-z0-9][a-z0-9-]{1,80}(?:/[0-9]+)?$")
FILE_PATTERN = r"^(TWO_WORLDS_ARTIFACTS[.]zip|two_worlds_job_receipt[.]json)$"
MAX_QUOTA_REPLY = 128_000
SAFE_FIELDS = ("used", "remaining", "total", "refreshAt", "refillAt", "period")
NUMBER_WITH_UNIT = re.compile(r"^[0-9]+(?:[.][0-9]+)?(?:h|m)?$")
REFRESH_DATE = re.compile(r"^20[0-9]{2}-[0-9TZ:+.\\-]{5,42}$")


class KaggleServiceError(RuntimeError):
    """Public-safe error, deliberately excludes Kaggle stderr and tokens."""


def _classify_cli_error(stderr: str, stdout: str = "") -> tuple[str, str]:
    # Never report untrusted remote error bodies. Only local, predefined enums.
    message = (stderr + "\n" + stdout).lower()[:64_000]
    if ("429" in message or "too many requests" in message
            or "rate limited" in message or "rate limit exceeded" in message):
        return "rate_limited", "HTTP 429 — stop requests; Kaggle rate limit is dynamic"
    if "401" in message or "unauthorized" in message or "invalid token" in message:
        return "authentication_problem", "Authentication or token rejected"
    if "403" in message or "forbidden" in message:
        return "forbidden", "Kaggle API forbidden or account restriction"
    if "404" in message or "not found" in message:
        return "not_found", "Kaggle API resource unavailable"
    return "unknown_api_failure", "Kaggle API request failed; no automatic retries"


def _execute(cmd: list[str], *, runner=subprocess.run, timeout: int = 90):
    try:
        result = runner(cmd, check=False, capture_output=True,
                        text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise KaggleServiceError("KAGGLE_API_TIMEOUT: one request timed out") from None
    except (OSError, ValueError):
        raise KaggleServiceError("KAGGLE_CLI_UNAVAILABLE: no request completed") from None
    if result.returncode != 0:
        code, detail = _classify_cli_error(str(result.stderr or ""), str(result.stdout or ""))
        raise KaggleServiceError("KAGGLE_" + code.upper() + ": " + detail)
    return result


def _clean_quota_row(row: dict) -> dict:
    # A string-valued duration like "21.94h" is not a REST request allowance.
    out = {}
    for key in SAFE_FIELDS:
        val = row.get(key)
        if key in {"used", "remaining", "total"} and isinstance(val, (int, float)):
            if not isinstance(val, bool) and 0 <= val <= 100_000:
                out[key] = val
        elif key in {"used", "remaining", "total"} and isinstance(val, str):
            if NUMBER_WITH_UNIT.fullmatch(val):
                out[key] = val
        elif key in {"refreshAt", "refillAt"} and isinstance(val, str):
            if len(val) <= 64 and REFRESH_DATE.fullmatch(val):
                out[key] = val
        elif key == "period" and isinstance(val, str) and val.lower() in {"daily", "weekly", "monthly"}:
            out[key] = val.lower()
    # The published CLI quota has separate GPU/TPU rows. Some versions use a
    # device type/accelerator field; only label known strings, never usernames.
    description = " ".join(str(row.get(x, ""))[:60].lower()
                           for x in ("name", "type", "resource", "accelerator", "quota"))
    if "gpu" in description:
        out["resource"] = "GPU"
    elif "tpu" in description:
        out["resource"] = "TPU"
    else:
        out["resource"] = "unlabeled_accelerator"
    return out


def parse_compute_quota(raw: str) -> dict:
    """Parse GPU/TPU compute-hours; report API rate-request limit as unpublished."""
    if len(raw) > MAX_QUOTA_REPLY:
        raise KaggleServiceError("KAGGLE_QUOTA_INVALID: oversized quota response")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        raise KaggleServiceError("KAGGLE_QUOTA_INVALID: not JSON") from None
    if isinstance(data, dict):
        data = data.get("quotas", data.get("result", data))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or len(data) > 30 or any(not isinstance(x, dict) for x in data):
        raise KaggleServiceError("KAGGLE_QUOTA_INVALID: unexpected response structure")
    return {
        "status": "quota_reported",
        "compute_quota_rows": [_clean_quota_row(row) for row in data],
        "kaggle_api_requests_per_minute": None,
        "kaggle_api_requests_per_day": None,
        "api_limit_status": "dynamic_undisclosed",
        "new_gpu_job_launched": False,
    }


def quota_once(*, runner=subprocess.run) -> dict:
    result = _execute(["kaggle", "quota", "--format", "json"],
                      runner=runner, timeout=60)
    return parse_compute_quota(result.stdout)


def output_once(kernel: str, target: Path, *, runner=subprocess.run) -> dict:
    """One filtered readback, not a loop. Fail closed on 429 for later recovery."""
    if not isinstance(kernel, str) or not KERNEL.fullmatch(kernel):
        raise ValueError("Expected safe Kaggle kernel slug")
    if not target.is_dir():
        raise FileNotFoundError("Output directory must exist before readback")
    # 200 metadata entries/page reduces pagination API calls. File filtering
    # still scans all pages; no assertion that a single CLI process is one HTTP call.
    args = ["kaggle", "kernels", "output", kernel, "-p", str(target),
            "--file-pattern", FILE_PATTERN, "--page-size", "200"]
    _execute(args, runner=runner, timeout=240)
    files = sorted(x.name for x in target.iterdir() if x.is_file()
                   and x.name in {"TWO_WORLDS_ARTIFACTS.zip", "two_worlds_job_receipt.json"})
    if len(files) != 2:
        raise KaggleServiceError("KAGGLE_OUTPUT_INCOMPLETE: expected ZIP and JSON receipt")
    return {"status": "two_private_proof_files_downloaded",
            "files": files, "api_cli_attempts": 1,
            "image_verified": False, "gpu_spent": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Safe Kaggle one-shot quota or private output")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("quota")
    out = sub.add_parser("output")
    out.add_argument("--kernel", required=True)
    out.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "quota":
            report = quota_once()
        else:
            report = output_once(args.kernel, args.output_dir)
    except KaggleServiceError as exc:
        # Sanitized error and no leaked remote body, account or token.
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
