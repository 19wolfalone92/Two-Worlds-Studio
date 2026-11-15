"""Regression tests: V8 cannot certify reused final decoded frames as unique acting."""
from studio.eb01_v8_40s_native_motion_plan import plan, verify_real_neural_sources


def _receipts():
    rows = {}
    counter = 1
    for scene in plan()["scenes"]:
        sid = scene["scene_id"]
        actor = scene["canonical_reference_sha256"]
        prev = None
        proofs = []
        for i in range(scene["native_segment_count"]):
            tail = f"{counter:064x}"
            proofs.append({
                "mp4_sha256": f"{counter + 1000:064x}",
                "conditioning_input_frame_sha256": prev or actor,
                "last_decoded_frame_sha256": tail,
                "canonical_actor_sha256": actor,
                "initial_reference_sha256": actor if i == 0 else None,
                "native_frames": 33,
                "native_fps": 24,
                "actual_model_inference": True,
                "rgb_neon_qc": "PASS",
                "identity_artistic_qc": "PASS",
            })
            prev = tail
            counter += 1
        rows[sid] = proofs
    return rows


def test_distinct_tails_allow_receipt_gate_but_not_movie_approval():
    result = verify_real_neural_sources(_receipts())
    assert result["status"] == "SEGMENT_QA_PASS_OWNER_REVIEW_STILL_REQUIRED"
    assert result["genuinely_original_segments_verified"] == 30
    assert result["full_40s_movie_ready"] is False


def test_reused_tail_from_another_scene_is_rejected():
    rows = _receipts()
    rows["S01"][0]["last_decoded_frame_sha256"] = rows["S00"][0]["last_decoded_frame_sha256"]
    result = verify_real_neural_sources(rows)
    assert result["status"] == "BLOCKED"
    assert any("REUSED_DECODED_TAIL" in r for r in result["reasons"])
    assert result["full_40s_movie_ready"] is False
