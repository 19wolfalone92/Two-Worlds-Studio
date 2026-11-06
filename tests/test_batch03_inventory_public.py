"""Offline tests for aggregate-only Kaggle inventory; no account network calls."""
import csv
import io
import os
import unittest
from unittest.mock import patch
from tools.batch03_inventory_public import discover_output_kinds, inspect, list_owned_kernels, main


class InventoryTests(unittest.TestCase):
    def test_detects_only_original_six_approved_archives(self):
        rows = [
            {"fileName": "TwoWorlds_PRIVATE_13_image.zip"},
            {"fileName": "/outputs/TwoWorlds_PRIVATE_14_voice.zip"},
            {"fileName": "TwoWorlds_PRIVATE_18_motion.zip"},
            {"fileName": "TwoWorlds_PRIVATE_99_image.zip"},
            {"fileName": "sensitive_narration.txt"},
        ]
        self.assertEqual(discover_output_kinds(rows),
                         {("13", "image"), ("14", "voice"), ("18", "motion")})

    def test_ignores_foreign_kernel_names_and_does_not_log_them(self):
        data = "ref,title\\nother/secret-notebook,secret\\nwolf19loner92/tw-batch03-image-20261104-01,private\\n"
        data = data.replace("\\n", "\n")
        with patch("tools.batch03_inventory_public.invoke", return_value=(True, data)):
            handles, ok = list_owned_kernels()
        self.assertTrue(ok)
        self.assertEqual(handles, ["wolf19loner92/tw-batch03-image-20261104-01"])

    def test_aggregates_distinct_stories_not_zip_file_count(self):
        def fake(*args, **kwargs):
            if args[:2] == ("kernels", "list"):
                return True, ("ref,title\n"
                              "wolf19loner92/tw-batch03-image-20261104-01,private\n"
                              "wolf19loner92/tw-batch03-voice-20261104-01,private\n")
            if args[:2] == ("kernels", "files"):
                if "image" in args[2]:
                    return True, ("fileName\nTwoWorlds_PRIVATE_13_image.zip\n"
                                  "TwoWorlds_PRIVATE_14_image.zip\n")
                return True, ("fileName\nTwoWorlds_PRIVATE_13_voice.zip\n"
                              "TwoWorlds_PRIVATE_16_voice.zip\n")
            raise AssertionError("Unexpected CLI invocation (mutating?)")
        with patch("tools.batch03_inventory_public.invoke", side_effect=fake):
            result = inspect()
        self.assertTrue(result["read_only"])
        self.assertEqual(result["image_zip_story_count"], 2)
        self.assertEqual(result["voice_zip_story_count"], 2)
        self.assertEqual(result["image_and_voice_zip_story_count"], 1)
        self.assertEqual(result["actual_rendered_mp4_verified"], 0)
        self.assertNotIn("secret", str(result))

    def test_missing_auth_does_not_proceed(self):
        with patch.dict(os.environ, {"KAGGLE_API_TOKEN": "", "KAGGLE_USERNAME": ""}):
            with patch("tools.batch03_inventory_public.inspect") as service:
                self.assertEqual(main(), 2)
                service.assert_not_called()
