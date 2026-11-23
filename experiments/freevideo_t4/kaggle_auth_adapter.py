"""Safely detect Kaggle CLI credential format inside the isolated Freevideo runner.

This adapter never prints token bytes/username and never writes secrets. It
accepts a current Kaggle access token or a legacy kaggle.json JSON string
stored in KAGGLE_API_TOKEN by an earlier workflow. No retries on API failure.
"""
from __future__ import annotations

import json
import os
import re
import subprocess

ERROR_LABELS = (
    ("rate_limited", ("429", "too many requests", "rate limit")),
    ("auth_required", ("authentication required", "you must be logged", "please log in", "not authenticated")),
    ("unauthorized", ("401", "unauthorized", "invalid api", "invalid token", "token expired")),
    ("forbidden", ("403", "forbidden")),
    ("not_found", ("404", "not found")),
    ("dns_or_network", ("name resolution", "failed to resolve", "dns", "connection refused", "connectionerror")),
    ("legacy_file_required", ("kaggle.json", "could not find kaggle")),
    ("cli_incompatible", ("unrecognized arguments", "invalid choice", "no such option")),
)


class KaggleAuthBlocked(RuntimeError):
    pass


def _legacy(raw: str) -> tuple[str, str] | None:
    if not raw.lstrip().startswith("{"):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise KaggleAuthBlocked("MALFORMED_SECRET_JSON") from exc
    if not isinstance(data, dict) or set(data) != {"username", "key"}:
        raise KaggleAuthBlocked("INVALID_LEGACY_SECRET_SCHEMA")
    username, key = data["username"], data["key"]
    if (not isinstance(username, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,29}", username)
            or not isinstance(key, str) or not 8 <= len(key) <= 256
            or any(ord(c) < 33 or ord(c) > 126 for c in key)):
        raise KaggleAuthBlocked("INVALID_LEGACY_SECRET_FIELDS")
    return username, key


def token_environment(env: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Normalize a modern token or legacy JSON secret in process memory."""
    raw = env.get("KAGGLE_API_TOKEN", "")
    if not isinstance(raw, str) or not raw.strip():
        raise KaggleAuthBlocked("MISSING_KAGGLE_API_SECRET")
    result = dict(env)
    if raw.lstrip().startswith("{"):
        username, key = _legacy(raw)
        result.pop("KAGGLE_API_TOKEN", None)
        result["KAGGLE_USERNAME"] = username
        result["KAGGLE_KEY"] = key
        return "legacy_json", result
    if "\n" in raw or "\r" in raw or len(raw) > 4096:
        raise KaggleAuthBlocked("INVALID_KAGGLE_TOKEN_FORMAT")
    return "modern_access_token", result


def diagnose_error(message: str) -> str:
    # Never expose untrusted CLI output: it might echo credentials.
    lower = message.lower()[:16000]
    for label, fragments in ERROR_LABELS:
        if any(fragment in lower for fragment in fragments):
            return label
    return "unclassified_cli_error"


def check_quota(env=None, runner=subprocess.run) -> dict:
    mode, safe_env = token_environment(dict(os.environ if env is None else env))
    try:
        result = runner(["kaggle", "quota", "--format", "json"], env=safe_env,
                        text=True, capture_output=True, timeout=65, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise KaggleAuthBlocked("KAGGLE_CLI_UNAVAILABLE_OR_TIMED_OUT") from exc
    if result.returncode:
        raise KaggleAuthBlocked("KAGGLE_" + diagnose_error((result.stderr or "") + "\n" + (result.stdout or "")))
    try:
        from studio.kaggle_api_limits import parse_compute_quota
        info = parse_compute_quota(result.stdout)
    except (ValueError, RuntimeError) as exc:
        raise KaggleAuthBlocked("KAGGLE_QUOTA_RESPONSE_INVALID") from exc
    return {"status": "API_AUTHENTICATED_QUOTA_READ_ONLY",
            "credential_format": mode, "gpu_quota_rows": sum(
                row.get("resource") == "GPU" for row in info["compute_quota_rows"]),
            "gpu_launched": False, "published": False}


def main() -> int:
    try:
        status = check_quota()
        code = 0
    except KaggleAuthBlocked as exc:
        status, code = {"status": "BLOCKED", "category": str(exc),
                        "gpu_launched": False, "published": False}, 2
    print(json.dumps(status, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
