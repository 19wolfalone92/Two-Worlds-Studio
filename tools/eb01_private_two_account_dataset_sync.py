"""Owner-authorized one-way PRIVATE source transfer for a second real Kaggle user.

Only two individually authenticated Kaggle API tokens stored as GitHub secrets.
Never publishes, never shares Kaggle logins, never copies into public GitHub,
and never pushes GPU notebooks. The owner's existing source is unchanged.
If the second user already has access to the original, perform NO transfer.
The fallback creates exactly ONE private, SHA-verified mirror of the 2 actor
portraits and 6 canon voice WAVs on the authorized second account.
This module deliberately cannot set a dataset Public or update unknown data.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path

from studio.eb01_v7_storyboard import SRC_SHA, VOICE_SHA
from tools.eb01_kaggle2_auth_probe import query, SOURCE_DATASET
from tools.eb01_kaggle_token_env import token_environment

MIRROR_PREFIX = "eb01-private-studio-inputs-v1"
NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,29}$")
FILE_SHA = {**{who + ".png": value for who, value in SRC_SHA.items()},
            **{shot + ".wav": value for shot, value in VOICE_SHA.items()}}
MAX_FILE_BYTES = 80 * 1024 * 1024
MAX_ARCHIVE_BYTES = 240 * 1024 * 1024


class SourceSyncBlocked(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_error(out: str) -> str:
    """Never emit raw Kaggle output, which may include private filenames."""
    low = (out or "").lower()
    if "403" in low or "forbidden" in low or "permission denied" in low:
        return "ACCESS_DENIED"
    if "401" in low or "unauthorized" in low:
        return "AUTH_DENIED"
    if "429" in low or "too many requests" in low:
        return "RATE_LIMITED"
    if "404" in low or "not found" in low:
        return "NOT_FOUND"
    if "timed out" in low or "connection reset" in low:
        return "NETWORK_TIMEOUT"
    return "UNCLASSIFIED_CLI_ERROR"


def kaggle(token: str, args: list[str], *, timeout: int = 120) -> str:
    if not token:
        raise SourceSyncBlocked("MISSING_ACCOUNT_TOKEN")
    if args[:2] != ["kaggle", "datasets"]:
        raise SourceSyncBlocked("UNAPPROVED_KAGGLE_COMMAND")
    if any(v in args for v in ("--public", "-u", "delete", "version")):
        raise SourceSyncBlocked("PUBLIC_OR_MUTATING_DATASET_COMMAND_BLOCKED")
    with token_environment(token) as env:
        try:
            proc = subprocess.run(args, env=env, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, check=False, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise SourceSyncBlocked("CLI_TIMEOUT") from None
        except OSError:
            raise SourceSyncBlocked("KAGGLE_CLI_UNAVAILABLE") from None
    if proc.returncode != 0:
        raise SourceSyncBlocked(classify_error((proc.stdout or "") + "\n" +
                                              (proc.stderr or "")))
    # Do not return stdout: it may include private media filenames.
    return "OK"


def validated_extract(archive: Path, stage: Path,
                      expected: dict[str, str] | None = None) -> dict:
    """Extract only canonical bytes; reject tricks, duplicate basenames and ZIP bombs."""
    expected = FILE_SHA if expected is None else expected
    stage.mkdir(mode=0o700, parents=True, exist_ok=False)
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise SourceSyncBlocked("PRIVATE_ARCHIVE_TOO_LARGE")
    discovered: dict[str, zipfile.ZipInfo] = {}
    try:
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                name = Path(item.filename)
                if (name.is_absolute() or ".." in name.parts or
                    item.filename.startswith("/") or
                    stat.S_ISLNK(item.external_attr >> 16)):
                    raise SourceSyncBlocked("UNSAFE_KAGGLE_ARCHIVE_ENTRY")
                basename = name.name
                if basename in expected:
                    if basename in discovered:
                        raise SourceSyncBlocked("AMBIGUOUS_CANONICAL_SOURCE")
                    if item.file_size > MAX_FILE_BYTES or item.file_size < 8:
                        raise SourceSyncBlocked("INVALID_CANONICAL_FILE_SIZE")
                    discovered[basename] = item
            if set(discovered) != set(expected):
                raise SourceSyncBlocked("INCOMPLETE_CANONICAL_SOURCE_DATASET")
            for name, item in discovered.items():
                dst = stage / name
                digest = hashlib.sha256()
                total = 0
                with z.open(item) as source, dst.open("xb") as sink:
                    for chunk in iter(lambda: source.read(1 << 20), b""):
                        total += len(chunk)
                        if total > MAX_FILE_BYTES:
                            raise SourceSyncBlocked("CANONICAL_FILE_OVERSIZE")
                        sink.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != expected[name]:
                    raise SourceSyncBlocked("CANONICAL_FILE_SHA_MISMATCH")
                dst.chmod(0o600)
    except (zipfile.BadZipFile, RuntimeError, OSError, EOFError):
        raise SourceSyncBlocked("BROKEN_PRIVATE_DATASET_ZIP") from None
    return {"sources_sha_verified": len(discovered),
            "canonical_source_files": len(expected)}


def privacy_metadata(path: Path) -> bool:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeError):
        return False
    # Kaggle's dataset metadata must EXPLICITLY affirm private visibility.
    return isinstance(raw, dict) and raw.get("isPrivate") is True


def verify_private_mirror(second_token: str, dest_ref: str,
                          expected: dict[str, str], tmp: Path) -> str:
    metadir = tmp / "meta"
    metadir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        kaggle(second_token,
               ["kaggle", "datasets", "metadata", dest_ref, "-p", str(metadir)],
               timeout=100)
    except SourceSyncBlocked as exc:
        return "MIRROR_ABSENT" if exc.code == "NOT_FOUND" else "MIRROR_METADATA_" + exc.code
    if not privacy_metadata(metadir / "dataset-metadata.json"):
        return "MIRROR_PRIVATE_ACL_NOT_PROVEN"
    destdir = tmp / "readback"
    destdir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        kaggle(second_token, ["kaggle", "datasets", "download", "-d",
                              dest_ref, "-p", str(destdir), "--force",
                              "--quiet"], timeout=300)
    except SourceSyncBlocked as exc:
        return "MIRROR_READBACK_" + exc.code
    archives = list(destdir.glob("*.zip"))
    if len(archives) != 1:
        return "MIRROR_READBACK_ZIP_MISSING"
    try:
        validated_extract(archives[0], tmp / "validated", expected)
    except SourceSyncBlocked as exc:
        return "MIRROR_READBACK_" + exc.code
    return "MIRROR_PRIVATE_SHA_VERIFIED"


def perform_one_authorized_private_sync(owner_token: str,
                                        second_token: str,
                                        root: Path) -> dict:
    report = {
        "schema": 1, "source_files_expected": len(FILE_SHA),
        "two_distinct_accounts": False,
        "owner_canonical_source_unchanged": True,
        "second_private_copy_verified": False,
        "original_account_shared_access": False,
        "public_source_copy": False,
        "github_media_artifact": False,
        "gpu_jobs_submitted": 0,
        "published": False,
        "full_movie_ready": False,
    }
    try:
        if not owner_token or not second_token:
            raise SourceSyncBlocked("BOTH_KAGGLE_SECRETS_REQUIRED")
        if owner_token == second_token:
            raise SourceSyncBlocked("TOKENS_MUST_DIFFER")
        status1, owner = query(owner_token)
        status2, second = query(second_token)
        if (status1 != "AUTHENTICATED_READ_ONLY" or
            status2 != "AUTHENTICATED_READ_ONLY" or
            not owner or not second or owner == second or
            not NAME.fullmatch(second)):
            raise SourceSyncBlocked("DISTINCT_AUTHENTICATED_ACCOUNTS_REQUIRED")
        report["two_distinct_accounts"] = True
        # A run-specific create-only slug prevents destructive conflicts even
        # when Kaggle responds 403 to metadata reads for nonexistent private
        # datasets. No existing dataset is modified, deleted or overwritten.
        run_id = os.environ.get("GITHUB_RUN_ID", "")
        if not re.fullmatch(r"[1-9][0-9]{5,20}", run_id):
            raise SourceSyncBlocked("UNIQUE_GITHUB_RUN_ID_REQUIRED")
        unique = hashlib.sha256(("eb01-mirror-" + run_id).encode()).hexdigest()[:12]
        target = second + "/" + MIRROR_PREFIX + "-" + unique
        # If the second user already sees the first private dataset, do
        # not duplicate sensitive originals.
        try:
            kaggle(second_token, ["kaggle", "datasets", "files", SOURCE_DATASET],
                   timeout=90)
            report["original_account_shared_access"] = True
            report["status"] = "ORIGINAL_PRIVATE_DATASET_SHARED_NO_COPY"
            return report
        except SourceSyncBlocked as exc:
            if exc.code not in ("ACCESS_DENIED", "NOT_FOUND"):
                raise SourceSyncBlocked("ORIGINAL_ACCESS_PROBE_" + exc.code) from None
        existing = verify_private_mirror(second_token, target, FILE_SHA,
                                         root / "precheck")
        if existing == "MIRROR_PRIVATE_SHA_VERIFIED":
            report["status"] = "EXISTING_OWNER_APPROVED_PRIVATE_MIRROR_VERIFIED"
            report["second_private_copy_verified"] = True
            return report
        if existing not in ("MIRROR_ABSENT", "MIRROR_METADATA_ACCESS_DENIED"):
            raise SourceSyncBlocked(existing)
        # A 403 on a *new run-unique slug* is ambiguous, but attempting
        # a create-only PRIVATE dataset is non-destructive. Never use the
        # version/update/delete routes. Creation conflicts still block.
        ownerdir = root / "download"
        ownerdir.mkdir(mode=0o700)
        kaggle(owner_token, ["kaggle", "datasets", "download", "-d",
                             SOURCE_DATASET, "-p", str(ownerdir),
                             "--force", "--quiet"], timeout=300)
        archives = list(ownerdir.glob("*.zip"))
        if len(archives) != 1:
            raise SourceSyncBlocked("OWNER_PRIVATE_SOURCE_ARCHIVE_NOT_FOUND")
        stage = root / "safe-stage"
        validated_extract(archives[0], stage)
        metadata = {
            "id": target,
            "title": "EB01 Private Studio Inputs V1 " + unique,
            "subtitle": "Private two-character and six original-voice render source archive",
            "description": "Owner-approved original media for a separate legitimate Kaggle collaborator. Private source only. Never public.",
            "licenses": [{"name": "copyright-authors"}],
            "isPrivate": True,
        }
        (stage / "dataset-metadata.json").write_text(
            json.dumps(metadata) + "\n", encoding="utf-8")
        os.chmod(stage / "dataset-metadata.json", 0o600)
        # Kaggle CLI creates datasets PRIVATE by default; --public is banned.
        kaggle(second_token, ["kaggle", "datasets", "create", "-p",
                              str(stage), "--quiet", "--dir-mode", "skip"],
               timeout=420)
        report["private_upload_attempted"] = True
        readback = verify_private_mirror(second_token, target, FILE_SHA,
                                         root / "postcheck")
        if readback != "MIRROR_PRIVATE_SHA_VERIFIED":
            # Some Kaggle personal tokens return HTTP 403 on metadata even
            # when they can read a dataset. Do NOT bypass the ACL gate:
            # verify all eight SHA values with the second token AND prove
            # the uninvited first account cannot read that mirror.
            from tools.eb01_private_two_account_mirror_readback import verify_readback
            independent = verify_readback(
                owner_token, second_token, run_id, root / "negative-control")
            if independent.get("status") != (
                "SECOND_PRIVATE_ACL_AND_ALL_EIGHT_SOURCE_SHAS_VERIFIED"):
                raise SourceSyncBlocked(
                    "NEW_PRIVATE_MIRROR_" + readback + "_SECOND_READBACK_" +
                    str(independent.get("status", "UNKNOWN"))[:64])
        report["status"] = "NEW_PRIVATE_MIRROR_SHA_ACL_VERIFIED"
        report["second_private_copy_verified"] = True
        return report
    except SourceSyncBlocked as exc:
        report["status"] = "BLOCKED_" + exc.code
        return report


def main() -> int:
    os.umask(0o077)
    if not os.environ.get("RUNNER_TEMP"):
        raise RuntimeError("Only ephemeral GitHub runner operation is allowed")
    base = Path(os.environ["RUNNER_TEMP"]).resolve()
    gh = Path(os.environ.get("GITHUB_WORKSPACE", "/__invalid__")).resolve()
    if base == gh or gh in base.parents or base in gh.parents:
        raise RuntimeError("Runner temporary storage is not isolated from GitHub checkout")
    owner = os.environ.get("EB01_OWNER_TOKEN", "")
    second = os.environ.get("EB01_SECOND_TOKEN", "")
    with tempfile.TemporaryDirectory(prefix="eb01-privatesync-", dir=base) as tmp:
        report = perform_one_authorized_private_sync(owner, second, Path(tmp))
    safe = base / "EB01_TWO_ACCOUNT_SOURCE_SAFE_REPORT.json"
    safe.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    safe.chmod(0o600)
    print("EB01_PRIVATE_TWO_ACCOUNT_DATASET_SYNC",
          json.dumps(report, sort_keys=True))
    return 0 if report["status"] in (
        "ORIGINAL_PRIVATE_DATASET_SHARED_NO_COPY",
        "EXISTING_OWNER_APPROVED_PRIVATE_MIRROR_VERIFIED",
        "NEW_PRIVATE_MIRROR_SHA_ACL_VERIFIED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
