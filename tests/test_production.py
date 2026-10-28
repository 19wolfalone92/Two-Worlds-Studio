"""Additional regression coverage for fully local staging."""
import datetime as dt
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from studio.comic_art import panel
from studio.production import (scheduled_episodes, srt_timestamp, subtitles,
                               produce_episode, inspect_mp4)
from studio.render import validate_episode


class ArtPipelineTests(unittest.TestCase):
    def test_art_scene_visuals_change_with_story(self):
        a = panel({"art":"earth_mirror"}, 180, 320)
        b = panel({"art":"fortress"}, 180, 320)
        self.assertEqual(a.size,(180,320))
        self.assertNotEqual(a.tobytes(),b.tobytes())

    def test_subtitles_timing(self):
        parts=[{"id":"A","duration":2.5,"caption":"First"},
               {"id":"B","duration":1.5,"caption":"Second"}]
        sub=subtitles(parts,10)
        self.assertIn("00:00:00,000 --> 00:00:02,500",sub)
        self.assertIn("00:00:02,500 --> 00:00:04,000",sub)
        self.assertEqual(srt_timestamp(3661.123),"01:01:01,123")

    def test_bad_scene_art_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                validate_episode({"scenes":[{"id":"S","duration":1,"effect":"none","palette":"earth","art":"unbounded"}]}, Path(d))

    def test_no_unapproved_schedule(self):
        with tempfile.TemporaryDirectory() as d:
            plan = Path(d)/"release.json"
            plan.write_text(json.dumps({"episodes":[{"id":"E01","episode_file":"episodes/E01_THE_REFLECTION.json","release_date":"2026-11-03","editorial_approved":False,"rights_approved":True}]}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"approvals"):
                scheduled_episodes(plan,dt.date(2026,11,3))
            self.assertEqual(scheduled_episodes(plan,dt.date(2026,11,4)),[])

    def test_missing_date_does_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            plan = Path(d)/"release.json"
            plan.write_text('{"episodes":[{"id":"E01","release_date":null}]}',encoding="utf-8")
            self.assertEqual(scheduled_episodes(plan,dt.date(2026,11,3)),[])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg/ffprobe required")
    def test_produce_review_package(self):
        with tempfile.TemporaryDirectory() as d:
            temp=Path(d)
            episode= temp / "sample.json"
            episode.write_text(json.dumps({
                "id":"E92","title":"TEST",
                "scenes":[{"id":"S1","duration":1.1,"palette":"earth",
                           "art":"earth_mirror","effect":"mirror",
                           "heading":"A MIRROR","caption":"An impossible image."}]
            }),encoding="utf-8")
            manifest=produce_episode(episode,temp/"exports",width=180,height=320,fps=6)
            out=temp/"exports"/"E92"
            self.assertEqual(manifest["episode"],"E92")
            self.assertEqual(manifest["publication_status"],"awaiting_human_review")
            self.assertTrue((out/"E92_preview.mp4").is_file())
            self.assertTrue((out/"E92.srt").is_file())
            self.assertTrue((out/"POSTING_PACKAGE.md").is_file())
            self.assertTrue((out/"manifest.json").is_file())
            self.assertNotIn("publish",manifest.keys())


if __name__=="__main__":
    unittest.main()
