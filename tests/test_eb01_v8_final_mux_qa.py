from studio.eb01_v8_final_mux_qa import verify_final_mux

def probe(frames=967):
    return {"streams": [
        {"codec_type":"video","codec_name":"h264","width":576,"height":1024,
         "avg_frame_rate":"24/1","r_frame_rate":"24/1"},
        {"codec_type":"audio","codec_name":"aac"}],
        "format":{"duration":str(frames/24)}}

def check(p=None, **kw):
    return verify_final_mux(p if p is not None else probe(),
        decoded_video_frames=kw.pop("decoded_video_frames",967),
        independent_scene_count=kw.pop("independent_scene_count",7),
        canonical_audio_sha_verified=kw.pop("canonical_audio_sha_verified",True),
        owner_final_artistic_review=kw.pop("owner_final_artistic_review",True))

def test_exact_valid_metadata():
    result=check()
    assert result["movie_ready"] and not result["neural_provenance_independently_proven"]

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
