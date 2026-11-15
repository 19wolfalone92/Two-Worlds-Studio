from studio.eb01_v8_final_mux_qa import verify_final_mux


def probe(frames=967):
    return {"streams": [
        {"codec_type":"video","codec_name":"h264","width":576,"height":1024,
         "avg_frame_rate":"24/1","r_frame_rate":"24/1",
         "nb_frames":str(frames),"duration":str(frames/24)},
        {"codec_type":"audio","codec_name":"aac","duration":str(frames/24)}],
        "format":{"duration":str(frames/24)}}


def check(p=None, **kw):
    return verify_final_mux(p if p is not None else probe(),
        decoded_video_frames=kw.pop("decoded_video_frames",967),
        independent_scene_count=kw.pop("independent_scene_count",7),
        canonical_audio_sha_verified=kw.pop("canonical_audio_sha_verified",True),
        owner_final_artistic_review=kw.pop("owner_final_artistic_review",True),
        original_neural_segments_verified=kw.pop("original_neural_segments_verified",False),
        six_line_lipsync_verified=kw.pop("six_line_lipsync_verified",False),
        private_durable_storage_verified=kw.pop("private_durable_storage_verified",False),
        native_576x1024_detail_verified=kw.pop("native_576x1024_detail_verified",False))


def test_exact_valid_metadata_still_needs_all_proofs():
    result=check()
    assert not result["movie_ready"]
    assert not result["neural_provenance_independently_proven"]
    assert not result["native_576x1024_detail_independently_proven"]
    assert "ORIGINAL_NEURAL_PROVENANCE_MISSING" in result["reasons"]
    assert "NATIVE_576X1024_DETAIL_UNVERIFIED" in result["reasons"]


def test_missing_owner_signoff_blocks():
    assert not check(owner_final_artistic_review=False)["movie_ready"]


def test_short_film_blocks():
    assert "NATIVE_FRAME_COUNT_MISMATCH" in check(decoded_video_frames=563)["reasons"]


def test_low_resolution_blocks():
    p=probe()
    p["streams"][0]["width"]=320
    assert "RESOLUTION_BELOW_TARGET" in check(p)["reasons"]


def test_bad_fps_blocks():
    p=probe()
    p["streams"][0]["avg_frame_rate"]="30/1"
    assert "NOT_NATIVE_24_FPS" in check(p)["reasons"]


def test_missing_voice_proof_blocks():
    assert "SIX_LOCKED_VOICES_NOT_PROVEN" in check(canonical_audio_sha_verified=False)["reasons"]


def test_missing_scene_blocks():
    assert "SEVEN_SCENES_NOT_PROVEN" in check(independent_scene_count=6)["reasons"]


def test_upscaled_320x576_cannot_be_approved_as_native_576x1024_detail():
    result=check(original_neural_segments_verified=True,
                 six_line_lipsync_verified=True,
                 private_durable_storage_verified=True,
                 owner_final_artistic_review=True,
                 native_576x1024_detail_verified=False)
    assert not result["movie_ready"]
    assert "NATIVE_576X1024_DETAIL_UNVERIFIED" in result["reasons"]


def test_all_explicit_proofs_required_for_ready():
    result=check(original_neural_segments_verified=True,
                 six_line_lipsync_verified=True,
                 private_durable_storage_verified=True,
                 owner_final_artistic_review=True,
                 native_576x1024_detail_verified=True)
    assert result["movie_ready"] is True
    assert result["native_576x1024_detail_independently_proven"] is True


def test_invalid_frame_count_never_crashes_or_passes():
    for invalid in (None, "967", True, float("nan")):
        result=check(decoded_video_frames=invalid)
        assert result["movie_ready"] is False
        assert "NATIVE_FRAME_COUNT_MISMATCH" in result["reasons"]
        assert "INVALID_FINAL_DURATION" in result["reasons"]


def test_invalid_format_never_crashes_or_passes():
    p=probe()
    p["format"]=None
    result=check(p)
    assert result["movie_ready"] is False
    assert "INVALID_FINAL_DURATION" in result["reasons"]


def test_truncated_encoded_stream_rejected_despite_valid_container():
    p=probe()
    p["streams"][0]["nb_frames"]="563"
    assert "ENCODED_VIDEO_FRAME_COUNT_MISMATCH" in check(p,
        original_neural_segments_verified=True,
        six_line_lipsync_verified=True,
        private_durable_storage_verified=True,
        native_576x1024_detail_verified=True)["reasons"]


def test_short_audio_stream_rejected():
    p=probe()
    p["streams"][1]["duration"]="3.0"
    assert "AUDIO_STREAM_DURATION_MISMATCH" in check(p)["reasons"]


def test_missing_stream_metadata_fails_closed():
    p=probe()
    del p["streams"][0]["nb_frames"]
    del p["streams"][1]["duration"]
    result=check(p)
    assert "ENCODED_VIDEO_FRAME_COUNT_MISSING" in result["reasons"]
    assert "AUDIO_STREAM_DURATION_MISSING" in result["reasons"]
