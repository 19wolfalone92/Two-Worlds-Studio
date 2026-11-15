"""Offline tests for the read-only Kaggle owner dataset visibility probe."""
from __future__ import annotations

import os
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.eb01_kaggle_owner_dataset_probe import (
    TARGET_DATASET, inspect_owner_dataset_visibility, parse_owner_dataset_csv,
    classify_owner_list_failure
)


class OwnerDatasetVisibilityProbeTests(unittest.TestCase):
    def test_exact_slug_found_without_leaking_dataset_title(self):
        raw = "ref,title,size\n" + TARGET_DATASET + ",PRIVATE TITLE MUST NOT LEAK,123\n"
        result = parse_owner_dataset_csv(raw)
        self.assertEqual(result["status"], "OWNER_DATASET_LIST_OK_EXACT_SLUG_FOUND")
        self.assertTrue(result["exact_dataset_visible"])
        self.assertNotIn("PRIVATE TITLE", str(result))

    def test_exact_slug_absent(self):
        result = parse_owner_dataset_csv("ref,title,size\nother/dataset,Other,1\n")
        self.assertEqual(result["status"], "OWNER_DATASET_LIST_OK_EXACT_SLUG_NOT_FOUND")
        self.assertFalse(result["exact_dataset_visible"])

    def test_empty_list_is_a_successful_no_match(self):
        result = parse_owner_dataset_csv("ref,title,size\n")
        self.assertFalse(result["exact_dataset_visible"])
        self.assertEqual(result["rows_examined"], 0)

    def test_missing_owner_credential_blocks_before_cli(self):
        with patch.dict(os.environ, {}, clear=True):
            def forbidden(*args, **kwargs):
                raise AssertionError("CLI must not run without owner credential")
            result = inspect_owner_dataset_visibility(runner=forbidden)
        self.assertEqual(result["status"], "OWNER_KAGGLE_CREDENTIAL_MISSING")
        self.assertTrue(result["read_only"])
        self.assertFalse(result["dataset_created"])

    def test_cli_403_is_sanitized_and_never_authorizes_creation(self):
        with patch.dict(os.environ, {"KAGGLE_API_TOKEN": "test-token"}):
            def fake(*args, **kwargs):
                return SimpleNamespace(returncode=1, stdout="private rows",
                                       stderr="403 Forbidden")
            result = inspect_owner_dataset_visibility(runner=fake)
        self.assertEqual(result["status"], "OWNER_DATASET_LIST_ACCESS_DENIED")
        self.assertNotIn("private rows", str(result))
        self.assertFalse(result["dataset_created"])

    def test_cli_timeout_is_sanitized(self):
        with patch.dict(os.environ, {"KAGGLE_API_TOKEN": "test-token"}):
            def fake(*args, **kwargs):
                raise subprocess.TimeoutExpired("kaggle", 75)
            result = inspect_owner_dataset_visibility(runner=fake)
        self.assertEqual(result["status"], "OWNER_DATASET_LIST_HELP_TIMEOUT")

    def test_list_no_results_is_a_safe_exact_absence_signal(self):
        self.assertEqual(classify_owner_list_failure("No results found for search term"),
                         "OWNER_DATASET_LIST_OK_EXACT_SLUG_NOT_FOUND")

    def test_list_permission_and_argument_errors_remain_distinct(self):
        self.assertEqual(classify_owner_list_failure("403 Forbidden"),
                         "OWNER_DATASET_LIST_ACCESS_DENIED")
        self.assertEqual(classify_owner_list_failure("unrecognized arguments: --csv"),
                         "OWNER_DATASET_LIST_CLI_ARGUMENT_ERROR")

    def test_cli_invocation_is_owner_only_and_read_only_without_search_filter(self):
        with patch.dict(os.environ, {"KAGGLE_API_TOKEN": "test-token"}):
            seen = {"calls": []}
            def fake(args, **kwargs):
                seen["calls"].append((args, kwargs))
                if args[-1] == "--help":
                    return SimpleNamespace(returncode=0,
                        stdout="--mine --page --csv", stderr="")
                return SimpleNamespace(returncode=0,
                    stdout="ref,title,size\\n" + TARGET_DATASET + ",x,1\\n",
                    stderr="")
            result = inspect_owner_dataset_visibility(runner=fake)
        self.assertEqual(result["status"], "OWNER_DATASET_LIST_OK_EXACT_SLUG_FOUND")
        self.assertTrue(result["scan_complete"])
        self.assertEqual(result["pages_scanned"], 1)
        self.assertEqual(seen["calls"][0][0], ["kaggle", "datasets", "list", "--help"])
        args, kwargs = seen["calls"][1]
        self.assertIn("--mine", args)
        self.assertIn("--page", args)
        self.assertIn("--csv", args)
        self.assertNotIn("--search", args)
        self.assertNotIn("create", args)
        self.assertNotIn("download", args)
        self.assertNotIn("kernels", args)
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)

    def test_private_owner_listing_is_paginated_until_exact_slug_found(self):
        with patch.dict(os.environ, {"KAGGLE_API_TOKEN": "test-token"}):
            pages = []
            def fake(args, **kwargs):
                if args[-1] == "--help":
                    return SimpleNamespace(returncode=0,
                        stdout="--mine --page --csv", stderr="")
                page = int(args[args.index("--page") + 1])
                pages.append(page)
                if page == 1:
                    rows = "".join(f"other/{i},title-{i},1\\n" for i in range(20))
                    return SimpleNamespace(returncode=0,
                        stdout="ref,title,size\\n" + rows, stderr="")
                return SimpleNamespace(returncode=0,
                    stdout="ref,title,size\\n" + TARGET_DATASET + ",private-title,1\\n",
                    stderr="")
            result = inspect_owner_dataset_visibility(runner=fake)
        self.assertEqual(pages, [1, 2])
        self.assertEqual(result["status"], "OWNER_DATASET_LIST_OK_EXACT_SLUG_FOUND")
        self.assertTrue(result["exact_dataset_visible"])
        self.assertTrue(result["scan_complete"])
        self.assertEqual(result["rows_examined"], 21)
        self.assertNotIn("private-title", str(result))

    def test_owner_listing_absence_is_reported_only_after_short_final_page(self):
        with patch.dict(os.environ, {"KAGGLE_API_TOKEN": "test-token"}):
            def fake(args, **kwargs):
                if args[-1] == "--help":
                    return SimpleNamespace(returncode=0,
                        stdout="--mine --page --csv", stderr="")
                rows = "ref,title,size\\nother/dataset,private-title,1\\n"
                return SimpleNamespace(returncode=0, stdout=rows, stderr="")
            result = inspect_owner_dataset_visibility(runner=fake)
        self.assertEqual(result["status"], "OWNER_DATASET_LIST_OK_EXACT_SLUG_NOT_FOUND")
        self.assertFalse(result["exact_dataset_visible"])
        self.assertTrue(result["scan_complete"])
        self.assertEqual(result["pages_scanned"], 1)
        self.assertNotIn("private-title", str(result))



if __name__ == "__main__":
    unittest.main()
