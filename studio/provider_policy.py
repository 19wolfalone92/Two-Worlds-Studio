"""Provider-neutral, fail-closed retry scheduling policy.

Shared by Kaggle, CPU workers and private SQLite queue. A planned retry is
NOT an automatic provider call, billing authorization or license override.
"""
from __future__ import annotations
from dataclasses import dataclass

SAFE_REASONS = frozenset({
    "http_429", "quota_exhausted", "network_error", "provider_unavailable",
    "dependency_missing", "invalid_response", "unsafe_input", "unknown_error",
})
TERMINAL = frozenset({"unsafe_input", "invalid_response", "dependency_missing"})
KAGGLE_ERRORS = {
    "KAGGLE_RATE_LIMITED": "http_429",
    "KAGGLE_QUOTA_EXHAUSTED": "quota_exhausted",
    "KAGGLE_API_TIMEOUT": "network_error",
    "KAGGLE_CLI_UNAVAILABLE": "dependency_missing",
    "KAGGLE_AUTHENTICATION_PROBLEM": "unsafe_input",
    "KAGGLE_FORBIDDEN": "unsafe_input",
    "KAGGLE_NOT_FOUND": "invalid_response",
}


@dataclass(frozen=True)
class RetryDecision:
    reason: str
    state: str
    retry_after_seconds: int
    external_requests_started: int = 0
    fallback_provider_selected: bool = False


def classify_kaggle_failure(error: BaseException) -> str:
    """Map stable internal error codes only; never expose raw HTTP text."""
    from studio.kaggle_api_limits import KaggleServiceError
    if not isinstance(error, KaggleServiceError):
        return "unknown_error"
    code = str(error).split(":", 1)[0]
    return KAGGLE_ERRORS.get(code, "unknown_error")


def decide_retry(reason: str, attempt: int, *,
                 max_attempts: int = 4, retry_after: int | None = None) -> RetryDecision:
    """Same bounded queue backoff, without fake fixed API rate-limit numbers."""
    if reason not in SAFE_REASONS:
        raise ValueError("Failure reason must be a predefined code, never raw remote text")
    if type(attempt) is not int or not 1 <= attempt <= 100:
        raise ValueError("Invalid attempt counter")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 100:
        raise ValueError("Invalid max attempts")
    if retry_after is not None and (type(retry_after) is not int or not 0 <= retry_after <= 86400):
        raise ValueError("Invalid Retry-After")
    if attempt >= max_attempts or reason in TERMINAL:
        return RetryDecision(reason, "failed", 0)
    delay = max(min(3600, 60 * 2 ** (attempt - 1)), retry_after or 0)
    if reason == "quota_exhausted":
        delay = max(delay, 86400)
    # The dynamic rate limit is deliberately not converted into fictitious
    # "requests/minute" or "safe to spend" authorization.
    return RetryDecision(reason, "waiting", delay)
