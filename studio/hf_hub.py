"""Read-only Hugging Face Hub preflight for TWO WORLDS Studio.

Checks GitHub Actions HF_TOKEN and reviewed model metadata. No weights,
paid inference, code execution, media approval or credentials are output.
"""
from __future__ import annotations

import argparse
import json
import os
from urllib import error, request

HUB_API = "https://huggingface.co/api"
REVIEWED_MODELS = {
    "image": {"repo": "black-forest-labs/FLUX.2-klein-4B", "expected_license": "apache-2.0"},
    "voice": {"repo": "hexgrad/Kokoro-82M", "expected_license": "apache-2.0"},
}


class HubPreflightError(RuntimeError):
    """Error safe to print without leaking credentials or response body."""


def hub_json(path: str, token: str, *, opener=request.urlopen) -> dict:
    """Access only the constant Hub API host with bounded responses."""
    if not path.startswith("/") or ".." in path or "?" in path or "#" in path:
        raise ValueError("Unsafe Hub API path")
    if not token or not token.startswith("hf_"):
        raise HubPreflightError("HF_TOKEN is missing or is not a Hugging Face token")
    req = request.Request(
        HUB_API + path,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with opener(req, timeout=15) as response:
            body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise HubPreflightError("Hub metadata response is too large")
        result = json.loads(body)
    except error.HTTPError as exc:
        raise HubPreflightError(f"Hugging Face API returned HTTP {exc.code}") from None
    except (error.URLError, TimeoutError, OSError) as exc:
        raise HubPreflightError(f"Cannot reach Hugging Face ({type(exc).__name__})") from None
    except (ValueError, UnicodeError):
        raise HubPreflightError("Hugging Face returned invalid JSON") from None
    if not isinstance(result, dict):
        raise HubPreflightError("Unexpected Hub response format")
    return result


def planned_report() -> dict:
    return {
        "mode": "dry-run",
        "authentication_tested": False,
        "models": [{"role": role, **spec} for role, spec in REVIEWED_MODELS.items()],
        "weights_downloaded": False,
        "inference_performed": False,
        "note": "Metadata/authentication only; no claim about GPU quota, weight access or production quality.",
    }


def check_connection(token: str, *, query=hub_json) -> dict:
    account = query("/whoami-v2", token)
    if not (isinstance(account.get("name"), str) and account["name"]):
        raise HubPreflightError("Hub authentication not confirmed")
    results = []
    for role, spec in REVIEWED_MODELS.items():
        repo = spec["repo"]
        metadata = query("/models/" + repo, token)
        license_name = (
            metadata.get("cardData", {}).get("license")
            if isinstance(metadata.get("cardData"), dict) else None
        )
        if metadata.get("id") != repo:
            raise HubPreflightError(f"Unexpected metadata for reviewed {role} model")
        if license_name != spec["expected_license"]:
            raise HubPreflightError(f"Review {role} model license before use (Hub metadata changed or absent)")
        results.append({
            "role": role,
            "repository": repo,
            "revision": metadata.get("sha") if isinstance(metadata.get("sha"), str) else None,
            "license": license_name,
            "gated": metadata.get("gated", False),
            "metadata_accessible": True,
            "weights_access_verified": False,
        })
    return {
        "status": "hub_authenticated_and_model_metadata_verified",
        "authentication_tested": True,
        "models": results,
        "weights_downloaded": False,
        "inference_performed": False,
        "gpu_allocation_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Hugging Face token and reviewed-model check")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Offline plan; does not read a token")
    group.add_argument("--check", action="store_true", help="Authenticate and check reviewed model metadata")
    args = parser.parse_args()
    try:
        result = planned_report() if args.dry_run else check_connection(os.environ.get("HF_TOKEN", ""))
    except HubPreflightError as exc:
        parser.exit(1, f"Hugging Face preflight failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
