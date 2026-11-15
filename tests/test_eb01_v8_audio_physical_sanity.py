"""Regression: EB01 audio QA must reject impossible motion and type spoofing."""
from studio.eb01_v8_audio_conditioned_qa import verify_audio_conditioned_performances
from studio.eb01_v8_40s_native_motion_plan import plan


def _valid_evidence():
    evidence = {}
    for scene in plan()["scenes"]:
        if scene["canonical_voice_sha256"] is None:
            continue
        sid = scene["scene_id"]
        evidence[sid] = {
            "scene_id": sid,
            "voice_sha256": scene["canonical_voice_sha256"],
            "dialogue_start_frame": scene["dialogue_start_scene_frame"],
            "dialogue_end_frame": scene["dialogue_end_scene_frame"],
            "decoded_scene_frames": scene["unique_movie_frames"],
            "original_neural_frames": True,
            "audio_visual_offset_ms": 0.0,
            "jaw_audio_correlation": 0.8,
            "voiced_jaw_motion_px": 2.0,
            "silent_jaw_motion_px": 0.2,
            "anatomical_artistic_owner_review": "PASS",
        }
    return evidence


def test_nominal_metadata_does_not_claim_movie_ready():
    result = verify_audio_conditioned_performances(_valid_evidence())
    assert result["verified_lines"] == 6
    assert result["movie_ready"] is False


def test_negative_jaw_displacement_rejected():
    evidence = _valid_evidence()
    evidence["S02"]["voiced_jaw_motion_px"] = -1.0
    evidence["S02"]["silent_jaw_motion_px"] = -2.0
    result = verify_audio_conditioned_performances(evidence)
    assert result["status"] == "BLOCKED"
    assert "S02_AUDIO_CONDITIONING_NOT_DEMONSTRATED" in result["reasons"]


def test_boolean_frame_count_rejected():
    evidence = _valid_evidence()
    evidence["S02"]["dialogue_start_frame"] = True
    assert verify_audio_conditioned_performances(evidence)["status"] == "BLOCKED"


def test_negative_silent_displacement_rejected():
    evidence = _valid_evidence()
    evidence["S02"]["silent_jaw_motion_px"] = -0.1
    assert verify_audio_conditioned_performances(evidence)["status"] == "BLOCKED"
