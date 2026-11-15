"""EB01 theatrical 40+ second plan: only ORIGINAL neural acting, no loops/holds.

The existing 7-shot V7 storyboard is 563 frames (23.458 s at 24fps).
It is mathematically incapable of delivering the approved >40s movie.
This V8 overlay preserves every original portrait/voice/prompt SHA and
dialogue, and schedules 30 genuine native 33-frame I2V takes. A single
conditioning frame may overlap the previous take, but it is included
ONCE in the final source timeline. No fake interpolated/copy-pasted frames.

Scheduling is NOT generation. Stage receipts and actual 24fps motion are
mandatory and owner artistic approval remains separate.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import re
from typing import Mapping

from studio.eb01_v7_storyboard import (
    SHOTS, SRC_SHA, VOICE_SHA, VOICE_SECONDS,
)

FPS = 24
SEGMENT_NATIVE_FRAMES = 33
SEGMENT_OVERLAP_FRAMES = 1
STRIDE = SEGMENT_NATIVE_FRAMES - SEGMENT_OVERLAP_FRAMES
REQUIRED_MIN_DURATION_SECONDS = 40.0

# Deliberate speaking/reaction intervals. 7 unique movie scenes, not
# additional recycled pseudo scenes. Natural acting beats after/before voice.
SEGMENT_COUNTS = {
    "S00": 3,  # 4.04s: interview room, skeleton preps bowtie and folder
    "S01": 4,  # 5.38s: Elizabeth reviews CV and says first question
    "S02": 6,  # 8.04s: Kostya introduces himself, anatomy and paper gesture
    "S03": 3,  # 4.04s: dry reaction, "Возраст?"
    "S04": 5,  # 6.71s: Kostya answers, shoulder shrug and settling
    "S05": 5,  # 6.71s: disbelief, eyebrow, pen, "А где душа?"
    "S06": 4,  # 5.38s: dry punchline, then earned audience reaction
}
INTENTIONAL_ACTION = {
    "S00": "4s original camera dolly, suit adjustment and skeletal hand",
    "S01": "real human eyes/cheeks, review paper, gentle shoulder lift",
    "S02": "moving bone jaw and hand gesture, speaking then a dry reaction",
    "S03": "real eyebrow, cheek muscles, subtle neck reaction and question",
    "S04": "jaw moves with voice, shoulders shrug and skeleton settles",
    "S05": "pen placement, eyebrow changes, eyes, human jaw/cheeks",
    "S06": "dry punchline, coherent bone jaw, hand lowers and final reaction",
}
HEX64 = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class SceneSchedule:
    scene_id: str
    actor_id: str
    canonical_reference_sha256: str
    canonical_voice_sha256: str | None
    native_segment_count: int
    native_frames_per_segment: int
    overlap_frames_between_segments: int
    unique_movie_frames: int
    start_movie_frame: int
    end_movie_frame: int  # exclusive
    dialogue_start_scene_frame: int | None
    dialogue_end_scene_frame: int | None
    duration_seconds: float
    original_v7_frames: int
    action_plan: str


def plan() -> dict:
    expected_ids=[x[0] for x in SHOTS]
    if list(SEGMENT_COUNTS)!=expected_ids or set(INTENTIONAL_ACTION)!=set(expected_ids):
        raise AssertionError("Original seven approved scenes cannot be reordered")
    schedules=[]
    cursor=0
    total_generated=0
    seen_actors=set()
    for sid,actor,original_frames,*_ in SHOTS:
        if not HEX64.fullmatch(SRC_SHA[actor]):
            raise ValueError("Canonical private original actor SHA contract missing")
        seen_actors.add(actor)
        n=SEGMENT_COUNTS[sid]
        if not 1<=n<=16:
            raise ValueError("Unsafe scene count")
        frames=n*STRIDE + SEGMENT_OVERLAP_FRAMES
        # Start dialogue after ~1/3rd of the natural non-speaking acting time.
        if sid=="S00":
            voice_sha=None
            voice_start=None
            voice_end=None
        else:
            voice_sha=VOICE_SHA[sid]
            if not HEX64.fullmatch(voice_sha):
                raise ValueError("Original Qwen Russian voice SHA missing")
            speaking_frames=math.ceil(VOICE_SECONDS[sid]*FPS)
            available=frames-speaking_frames
            if available < 12:
                raise ValueError("No natural acting space for the original Russian line")
            voice_start=round(available/3)
            voice_end=voice_start+speaking_frames
            if voice_end>frames:
                raise AssertionError("Voice spills into another scene")
        schedules.append(SceneSchedule(
            scene_id=sid,
            actor_id=actor,
            canonical_reference_sha256=SRC_SHA[actor],
            canonical_voice_sha256=voice_sha,
            native_segment_count=n,
            native_frames_per_segment=SEGMENT_NATIVE_FRAMES,
            overlap_frames_between_segments=SEGMENT_OVERLAP_FRAMES,
            unique_movie_frames=frames,
            start_movie_frame=cursor,
            end_movie_frame=cursor+frames,
            dialogue_start_scene_frame=voice_start,
            dialogue_end_scene_frame=voice_end,
            duration_seconds=round(frames/FPS,4),
            original_v7_frames=original_frames,
            action_plan=INTENTIONAL_ACTION[sid],
        ))
        total_generated+=n*SEGMENT_NATIVE_FRAMES
        cursor+=frames
    if seen_actors!=set(SRC_SHA):
        raise AssertionError("Both approved canonical actors required")
    if cursor/FPS<REQUIRED_MIN_DURATION_SECONDS:
        raise AssertionError("No more 23-second false full film")
    if len(schedules)!=7 or sum(x.native_segment_count for x in schedules)!=30:
        raise AssertionError("Storyboard must contain 30 original neural takes")
    return {
        "schema":8,
        "required_fps":FPS,
        "new_true_inference_segment_count":30,
        "neural_frames_to_generate":total_generated,
        "final_unique_native_frames":cursor,
        "final_duration_seconds":round(cursor/FPS,4),
        "original_v7_native_frames":sum(x[2] for x in SHOTS),
        "original_v7_duration_seconds":round(sum(x[2] for x in SHOTS)/FPS,4),
        "original_character_actors":2,
        "original_russian_qwen_lines":6,
        "independent_scenes":len(schedules),
        "original_dialogue_preserved":True,
        "fake_frozen_frames_allowed":False,
        "looped_take_reuse_allowed":False,
        "public_publishing":False,
        "production_executed":False,
        "cinematic_owner_approved":False,
        "scenes":[asdict(x) for x in schedules],
    }


def verify_real_neural_sources(evidence: Mapping[str,list[dict]]) -> dict:
    """Reject fake scenes, same-source repeats or unproved 1-frame continuity.

    Needs SHA-authentic 33-frame native MP4 receipts, source input SHA and
    exact conditioning input-frame SHA and last decoded frame SHA for each
    GPU result. First decoded frame need NOT match input due VAE. Plan passes
    NOTHING. Only S00-S06 actors and native original 33f takes are valid.
    """
    p=plan()
    reasons=[]
    if not isinstance(evidence, Mapping):
        return {"schema": 8, "status": "BLOCKED",
                "reasons": ["INVALID_EVIDENCE_MAPPING"],
                "genuinely_original_segments_verified": 0,
                "expected_original_segments": p["new_true_inference_segment_count"],
                "full_40s_movie_ready": False, "social_published": False}
    if set(evidence)!=set(SEGMENT_COUNTS):
        reasons.append("REQUIRED_7_UNIQUE_SCENE_EVIDENCE_MISSING")
    sha_seen=set()
    decoded_tails_seen=set()
    genuine_count=0
    for row in p["scenes"]:
        sid=row["scene_id"]
        receipts=evidence.get(sid,[])
        if not isinstance(receipts,list) or len(receipts)!=row["native_segment_count"]:
            reasons.append(f"{sid}_MISSING_DISTINCT_NATIVE_I2V_SEGMENTS")
            continue
        prev_tail=None
        chain_valid=True
        for idx,proof in enumerate(receipts):
            if not isinstance(proof,dict):
                reasons.append(f"{sid}_{idx}_NO_GPU_RECEIPT")
                chain_valid=False
                continue
            sha=proof.get("mp4_sha256")
            # The VAE's first OUTPUT frame may differ pixel-by-pixel from
            # the input reference. Exact overlap must be enforced on the
            # conditioning SOURCE file SHA, not decoded output RGB bytes.
            first=proof.get("conditioning_input_frame_sha256")
            last=proof.get("last_decoded_frame_sha256")
            actor_sha=proof.get("canonical_actor_sha256")
            if (not all(isinstance(x,str) and HEX64.fullmatch(x)
                        for x in (sha,first,last,actor_sha))
                or actor_sha!=row["canonical_reference_sha256"]
                or sha in sha_seen
                or type(proof.get("native_frames")) is not int
                or proof["native_frames"] != 33
                or type(proof.get("native_fps")) is not int
                or proof["native_fps"] != 24
                or proof.get("actual_model_inference") is not True
                or proof.get("rgb_neon_qc")!="PASS"
                or proof.get("identity_artistic_qc")!="PASS"):
                reasons.append(f"{sid}_{idx}_REAL_NATIVE_GPU_OR_QA_NOT_PROVEN")
                chain_valid=False
                continue
            chain_ok = chain_valid
            if idx == 0:
                if proof.get("initial_reference_sha256") != row["canonical_reference_sha256"]:
                    reasons.append(f"{sid}_ORIGINAL_CHARACTER_SHA_MISMATCH")
                    chain_ok = False
            elif prev_tail is None or first != prev_tail:
                reasons.append(f"{sid}_{idx}_CONDITIONING_FRAME_CONTINUITY_BROKEN")
                chain_ok = False
            if last in decoded_tails_seen:
                reasons.append(f"{sid}_{idx}_REUSED_DECODED_TAIL")
                chain_ok = False
            sha_seen.add(sha)
            decoded_tails_seen.add(last)
            prev_tail = last
            chain_valid = chain_ok
            if chain_ok:
                genuine_count += 1
    return {
        "schema":8,
        "status":"SEGMENT_QA_PASS_OWNER_REVIEW_STILL_REQUIRED" if not reasons else "BLOCKED",
        "reasons":sorted(set(reasons)),
        "genuinely_original_segments_verified":genuine_count,
        "expected_original_segments":p["new_true_inference_segment_count"],
        "full_40s_movie_ready":False,  # no stitched file, owner or lipsync approval
        "social_published":False,
    }



def authorize_v8_scale_up(pilot: Mapping[str, object]) -> dict:
    """Fail closed BEFORE spending 30 GPU segments on an unapproved pilot.

    Optical luma differences and RGB QA cannot certify visible acting.
    The human review must refer to the exact SHA-locked private MP4, not
    an unrelated preview. Private durable storage must be access-controlled.
    This gate authorizes planning only, never GPU submission or publishing.
    """
    expected_sha = "74ee559ecf5d73cd95e0d28e0aa5171e6b546f6e227091974b42126acfea56d6"
    if not isinstance(pilot, Mapping):
        return {"schema": 8, "scale_up_authorized": False,
                "blocking_gates": ["INVALID_PILOT_EVIDENCE"],
                "gpu_job_submitted": False, "full_movie_ready": False,
                "public_media_export": False}
    checks = {
        "authentic_pilot_sha": pilot.get("pilot_mp4_sha256") == expected_sha,
        "true_native_33_frames": type(pilot.get("decoded_frames")) is int and pilot["decoded_frames"] == 33,
        "native_rgb_pass": pilot.get("rgb_status") == "NO_NEON_ARTIFACT_FOUND",
        "private_owner_visual_signoff": pilot.get("owner_visual_review") == "PASS",
        "skeletal_jaw_and_body_physics": pilot.get("skeletal_anatomy_review") == "PASS",
        "human_face_identity_and_expression": pilot.get("human_identity_review") == "PASS",
        "visible_nontrivial_acting": pilot.get("acting_motion_review") == "PASS",
        "private_durable_media_acl": pilot.get("private_media_acl_verified") is True,
        "audio_conditioned_lipsync_demonstrated": pilot.get("audio_conditioned_lipsync_review") == "PASS",
    }
    # No numeric luma threshold is an acceptable substitute for human QA.
    blocked = sorted(k for k, ok in checks.items() if not ok)
    return {
        "schema": 8,
        "scale_up_authorized": not blocked,
        "blocking_gates": blocked,
        "gpu_job_submitted": False,
        "full_movie_ready": False,
        "public_media_export": False,
    }

def main():
    print(json.dumps(plan(),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
