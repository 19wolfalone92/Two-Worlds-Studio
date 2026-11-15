"""Regression tests for V8 final acceptance gates; no private media involved."""
from studio.eb01_v8_final_mux_qa import verify_final_mux
from studio.eb01_v8_40s_native_motion_plan import plan

def _probe():
    return {"streams":[
        {"codec_type":"video","width":576,"height":1024,"avg_frame_rate":"24/1","r_frame_rate":"24/1","codec_name":"h264"},
        {"codec_type":"audio","codec_name":"aac"}],
        "format":{"duration":str(plan()["final_unique_native_frames"]/24)}}

def _verify(**kwargs):
    return verify_final_mux(_probe(),decoded_video_frames=plan()["final_unique_native_frames"],
        independent_scene_count=7,canonical_audio_sha_verified=True,
        owner_final_artistic_review=True,**kwargs)

def test_mux_metadata_alone_never_means_movie_ready():
    result=_verify()
    assert result["movie_ready"] is False
    assert {"ORIGINAL_NEURAL_PROVENANCE_MISSING","SIX_LINE_LIPSYNC_MISSING",
            "PRIVATE_DURABLE_STORAGE_UNVERIFIED"} <= set(result["reasons"])

def test_every_gate_is_required_independently():
    gates=("original_neural_segments_verified","six_line_lipsync_verified",
           "private_durable_storage_verified")
    for omitted in gates:
        args={key:True for key in gates if key!=omitted}
        assert _verify(**args)["movie_ready"] is False

def test_all_gates_explicitly_pass():
    result=_verify(original_neural_segments_verified=True,
        six_line_lipsync_verified=True,private_durable_storage_verified=True)
    assert result["movie_ready"] is True
    assert result["neural_provenance_independently_proven"] is True

def test_truthy_non_boolean_cannot_bypass_gates():
    result=_verify(original_neural_segments_verified=1,
        six_line_lipsync_verified="PASS",private_durable_storage_verified=True)
    assert result["movie_ready"] is False
