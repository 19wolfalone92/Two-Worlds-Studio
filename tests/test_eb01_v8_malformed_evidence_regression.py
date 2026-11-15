"""Regression tests for malformed V8 evidence; no private media involved."""
from studio.eb01_v8_40s_native_motion_plan import authorize_v8_scale_up, verify_real_neural_sources


def test_non_mapping_receipts_fail_closed():
    for value in (None, [], "not a mapping", 17):
        result = verify_real_neural_sources(value)
        assert result["status"] == "BLOCKED"
        assert result["genuinely_original_segments_verified"] == 0
        assert result["full_40s_movie_ready"] is False


def test_non_mapping_pilot_fails_closed():
    for value in (None, [], "bad", 17):
        result = authorize_v8_scale_up(value)
        assert result["scale_up_authorized"] is False
        assert result["gpu_job_submitted"] is False


def test_boolean_native_frame_count_cannot_authorize():
    result = authorize_v8_scale_up({"decoded_frames": True})
    assert result["scale_up_authorized"] is False
    assert "true_native_33_frames" in result["blocking_gates"]


def test_missing_scene_receipts_fail_closed():
    result = verify_real_neural_sources({})
    assert result["status"] == "BLOCKED"
    assert result["genuinely_original_segments_verified"] == 0
