"""Derive safe Kaggle account handle from an already configured personal API token.

Never prints, stores or sends the token to the user. GitHub Actions uses this
as a convenience when the optional KAGGLE_USERNAME variable is not configured.
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
from pathlib import Path

HANDLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,29}$")
LINE = re.compile(r"(?im)^\s*-?\s*username\s*:\s*([A-Za-z0-9_-]{2,30})\s*$")


def parse_username(config_output: str) -> str:
    matched = LINE.search(config_output)
    if matched is None or not HANDLE.fullmatch(matched.group(1)):
        raise ValueError("Could not determine Kaggle username from authenticated CLI config")
    return matched.group(1)


def resolve_username(explicit: str | None = None, *,
                     runner=subprocess.run) -> str:
    if explicit and explicit.strip():
        if not HANDLE.fullmatch(explicit.strip()):
            raise ValueError("Invalid provided Kaggle username")
        return explicit.strip()
    environment = os.environ.copy()
    environment.pop("KAGGLE_USERNAME", None)
    outcome = runner(["kaggle", "config", "view"], env=environment,
                     capture_output=True, text=True, timeout=60, check=True)
    return parse_username(outcome.stdout)


def main():
    p = argparse.ArgumentParser(description="Resolve account name from Kaggle token without exposing it")
    p.add_argument("--github-env", type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("GitHub secret KAGGLE_API_TOKEN is empty")
    name = resolve_username(os.environ.get("KAGGLE_USERNAME"))
    if args.github_env != Path(os.environ.get("GITHUB_ENV", "")):
        raise ValueError("Only the active GitHub Actions env file may be written")
    with args.github_env.open("a", encoding="utf-8") as f:
        f.write(f"KAGGLE_USERNAME={name}\n")
    print("PASS: Kaggle username resolved (account value redacted).")


if __name__ == "__main__":
    main()
