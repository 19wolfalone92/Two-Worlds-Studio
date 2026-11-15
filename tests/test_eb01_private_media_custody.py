"""No real media or credentials needed for custody-gate regression tests."""
import unittest

from studio.eb01_private_media_custody import validate_private_media_custody

SHA = "74ee559ecf5d73cd95e0d28e0aa5171e6b546f6e227091974b42126acfea56d6"


def good():
    return dict(backend="private_owner_object_store", visibility="private",
                owner_acl_verified=True, anonymous_access_denied=True,
                owner_readback_verified=True, retention_verified=True,
                public_artifact_export=False, mp4_sha256=SHA, byte_count=4096)


class PrivateMediaCustodyTests(unittest.TestCase):
    def test_verified_metadata_passes_but_not_remote_proof(self):
        result = validate_private_media_custody(good())
        self.assertTrue(result["media_ready"])
        self.assertFalse(result["remote_acl_independently_proven_by_this_function"])

    def test_missing_or_unsafe_claims_block(self):
        mutations = [
            ("visibility", "public"), ("owner_acl_verified", False),
            ("anonymous_access_denied", False), ("owner_readback_verified", False),
            ("retention_verified", False), ("public_artifact_export", True),
            ("backend", "public_github_artifact"), ("mp4_sha256", "bad"),
            ("byte_count", 0), ("byte_count", True),
        ]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                item = good()
                item[field] = value
                self.assertFalse(validate_private_media_custody(item)["media_ready"])

    def test_secret_fields_never_accepted(self):
        for field in ("signed_url", "api_key", "mp4_bytes", "raw_prompt"):
            with self.subTest(field=field):
                item = good()
                item[field] = "PRIVATE"
                self.assertFalse(validate_private_media_custody(item)["media_ready"])

    def test_empty_record_blocks(self):
        self.assertFalse(validate_private_media_custody({})["media_ready"])


if __name__ == "__main__":
    unittest.main()
