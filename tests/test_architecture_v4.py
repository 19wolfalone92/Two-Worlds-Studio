"""Offline integration assertions: NO GPU, network, publication or real voices."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.cli import doctor, run_engine, main as cli_main
from studio.voice_engine import VoiceRequest, route, execute
from studio.quality_hub import assess

ROOT = Path(__file__).resolve().parents[1]

try:
    import pydantic
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


class UnifiedEntrypointTests(unittest.TestCase):
    def test_doctor_reports_presence_not_secrets_or_live_quota(self):
        with patch.dict(os.environ, {"KAGGLE_API_TOKEN": "DO_NOT_LEAK_TOKEN"}):
            report = doctor()
        self.assertTrue(report["connections_configured_not_verified"]["KAGGLE_API_TOKEN"])
        self.assertNotIn("DO_NOT_LEAK_TOKEN", json.dumps(report))
        self.assertFalse(report["network_used"])
        self.assertFalse(report["published"])

    def test_plan_does_not_execute_or_echo_source_script(self):
        with patch("studio.cli.subprocess.run") as run:
            report = run_engine("qwen", ["--text", "private sentence"])
            run.assert_not_called()
        self.assertEqual(report["status"], "planned")
        self.assertNotIn("private sentence", json.dumps(report))
        self.assertFalse(report["actual_execution"])

    def test_engine_execution_gated_and_no_shell(self):
        with patch("studio.cli.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "allow-provider-usage"):
                run_engine("comfy", [], execute=True)
            run.assert_not_called()
            result = run_engine("qwen", ["--", "--dry-run"], execute=True,
                                allow_provider_usage=True)
            self.assertEqual(result["status"], "driver_exited_zero")
            self.assertEqual(run.call_args.kwargs["check"], True)
            self.assertEqual(run.call_args.args[0][-1], "--dry-run")

    def test_kaggle_quota_defaults_offline(self):
        with patch("studio.kaggle_api_limits.quota_once") as live:
            with patch("sys.stdout", new_callable=io.StringIO) as out:
                self.assertEqual(cli_main(["kaggle-quota"]), 0)
            live.assert_not_called()
            self.assertEqual(json.loads(out.getvalue())["api_limit_status"], "dynamic_undisclosed")


class VoiceRoutingTests(unittest.TestCase):
    def test_kokoro_and_qwen_share_opaque_routing_not_generated_audio(self):
        for speaker,lang,expected in [("Mira","en","kokoro"),("Mira","ru","qwen"),
                                      ("Elizabeth","ru","qwen"),("Kostya","ru","qwen")]:
            request=VoiceRequest(speaker, "Привет." if lang == "ru" else "Hello.", lang)
            with patch("studio.kokoro_tts.synthesize") as kokoro, patch("studio.russian_qwen_tts.synthesize") as qwen:
                info = route(request)
                kokoro.assert_not_called()
                qwen.assert_not_called()
            self.assertEqual(info["engine"], expected)
            self.assertFalse(info["synthesis_executed"])
            self.assertEqual(info["review_status"],"unreviewed")
            self.assertNotIn(request.text, json.dumps(info,ensure_ascii=False))

    def test_reject_unregistered_and_invalid_text(self):
        with self.assertRaisesRegex(ValueError, "registry"):
            route(VoiceRequest("Unknown", "Hello!", "en"))
        with self.assertRaisesRegex(ValueError, "550"):
            VoiceRequest("Mira", "A"*551, "en")

    def test_voice_output_must_be_outside_public_source_tree(self):
        with self.assertRaisesRegex(ValueError,"outside"):
            execute(VoiceRequest("Mira","Hello!","en"),ROOT/"dist"/"bad.wav")

    def test_voice_output_cannot_overwrite_an_existing_private_file(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/"clip.wav"
            target.write_bytes(b"original")
            with patch("studio.kokoro_tts.synthesize") as synth:
                with self.assertRaises(FileExistsError):
                    execute(VoiceRequest("Mira","Hello!","en"),target)
                synth.assert_not_called()


class UnifiedQualityTests(unittest.TestCase):
    def test_technical_pass_is_not_human_release_approval(self):
        with patch("studio.release_gate.check_video", return_value={"status":"technical_pass"}):
            result=assess(Path("example.mp4"),Path("example.srt"),Path("cover.png"))
        self.assertEqual(result["status"],"technical_pass_human_review_required")
        self.assertFalse(result["publication_authorized"])
        self.assertTrue(result["human_review_required"])

    def test_any_qc_failure_blocks_without_private_paths(self):
        with patch("studio.release_gate.check_video",side_effect=ValueError("/private/secret/frame.mp4")):
            result=assess(Path("example.mp4"),Path("example.srt"),Path("cover.png"))
        self.assertEqual(result["status"],"blocked")
        self.assertNotIn("/private/secret",json.dumps(result))

    def test_master_requires_source_episode(self):
        with self.assertRaisesRegex(ValueError,"source episode"):
            assess(Path("example.mp4"),Path("example.srt"),Path("cover.png"),master=True)


@unittest.skipUnless(HAS_PYDANTIC, "Install optional [validation] dependency to run contract tests")
class TypedContractsTests(unittest.TestCase):
    def test_existing_bible_and_fixtures_validate_without_mutation(self):
        from studio.contracts import validate_document, validate_bundle
        paths=[
          ("episode",ROOT/"episodes/E01_THE_REFLECTION.json"),
          ("visual_bible",ROOT/"story/visual_bible.json"),
          ("identities",ROOT/"story/identity_registry.json"),
        ]
        for kind,p in paths:
            before=p.read_bytes()
            result=validate_document(p,kind)
            self.assertEqual(result["status"],"validated")
            self.assertFalse(result["review_approved"])
            self.assertEqual(p.read_bytes(),before)
        ep=json.loads(paths[0][1].read_text())
        vb=json.loads(paths[1][1].read_text())
        ident=json.loads(paths[2][1].read_text())
        self.assertEqual(validate_bundle(ep,vb,ident)["status"],"validated")

    def test_duplicate_scene_id_and_bad_duration_fail_fast(self):
        from studio.contracts import Episode
        ep=json.loads((ROOT/"episodes/E01_THE_REFLECTION.json").read_text())
        ep["scenes"].append(ep["scenes"][0].copy())
        with self.assertRaisesRegex(ValueError,"Duplicate scene ID"):
            Episode.model_validate(ep)
        ep["scenes"].pop()
        ep["scenes"][0]["duration"]=-2
        with self.assertRaises(ValueError):
            Episode.model_validate(ep)

    def test_unknown_body_and_visual_location_fail_when_crosschecking(self):
        from studio.contracts import validate_bundle
        ep=json.loads((ROOT/"episodes/E01_THE_REFLECTION.json").read_text())
        vb=json.loads((ROOT/"story/visual_bible.json").read_text())
        ident=json.loads((ROOT/"story/identity_registry.json").read_text())
        ep["scenes"][0]["palette"]="not_a_real_location"
        with self.assertRaisesRegex(ValueError,"unknown visual location"):
            validate_bundle(ep,vb,ident)


if __name__ == "__main__":
    unittest.main()
