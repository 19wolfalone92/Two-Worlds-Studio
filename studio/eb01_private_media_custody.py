"""Private EB01 media custody gate. No media bytes or signed URLs in public logs.

This module verifies *metadata only*. It never uploads or downloads originals.
A storage backend must separately prove ACLs and retention before production.
"""
from __future__ import annotations

import re
from typing import Mapping

SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED = frozenset({"private_kaggle_dataset", "private_owner_object_store"})
FORBIDDEN_KEYS = frozenset({
    "url", "download_url", "signed_url", "token", "api_key", "password",
    "raw_prompt", "portrait", "voice_bytes", "mp4_bytes", "public_url",
})


def validate_private_media_custody(record: Mapping[str, object]) -> dict:
    """Fail closed on unknown ACL, unverified durability, or public exposure.

    A positive result is a metadata preflight, NOT proof of remote ACLs:
    independently verified owner credentials and an actual private readback
    must supply the asserted checks.
    """
    reasons = []
    if not isinstance(record, Mapping):
        return {"status": "BLOCKED", "reasons": ["INVALID_RECORD"], "media_ready": False}
    if FORBIDDEN_KEYS.intersection(record):
        reasons.append("SECRET_OR_RAW_MEDIA_FIELD_PRESENT")
    if record.get("backend") not in ALLOWED:
        reasons.append("UNAPPROVED_PRIVATE_BACKEND")
    if record.get("visibility") != "private":
        reasons.append("MEDIA_NOT_PRIVATE")
    if record.get("owner_acl_verified") is not True:
        reasons.append("OWNER_ACL_NOT_VERIFIED")
    if record.get("anonymous_access_denied") is not True:
        reasons.append("ANONYMOUS_ACCESS_NOT_DENIED")
    if record.get("owner_readback_verified") is not True:
        reasons.append("PRIVATE_OWNER_READBACK_NOT_VERIFIED")
    if record.get("retention_verified") is not True:
        reasons.append("DURABLE_RETENTION_NOT_VERIFIED")
    if record.get("public_artifact_export") is not False:
        reasons.append("PUBLIC_EXPORT_NOT_DISABLED")
    sha = record.get("mp4_sha256")
    if not isinstance(sha, str) or SHA256.fullmatch(sha) is None:
        reasons.append("ORIGINAL_VIDEO_SHA_MISSING")
    if not isinstance(record.get("byte_count"), int) or isinstance(record.get("byte_count"), bool) or record["byte_count"] < 1024:
        reasons.append("ORIGINAL_VIDEO_SIZE_UNVERIFIED")
    return {"status": "METADATA_PREFLIGHT_PASS" if not reasons else "BLOCKED",
            "reasons": sorted(reasons), "media_ready": not reasons,
            "remote_acl_independently_proven_by_this_function": False,
            "public_media_export": False}
