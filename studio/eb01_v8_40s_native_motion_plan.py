"""EB01 theatrical 40+ second plan: only ORIGINAL neural acting, no loops/holds.

The existing 7-shot V7 storyboard is 563 frames (23.458 s at 24fps).
It is mathematically incapable of delivering the approved >40s movie.
This V8 overlay preserves every original portrait/voice/prompt SHA and
dialogue, and schedules 30 genuine native 33-frame I2V takes. A single
conditioning frame may overlap the previous take, but it is included
ONCE in the final source timeline. No fake interpolated/copy-pasted frames.

Scheduling is NOT generation. Stage receipts and actual 24fps motion are
mandatory and owner artistic approval remains separate. Final source takes
must contain native >=576x1024 model detail; upscaling a 320x576 pilot is
not accepted as cinematic native detail.
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
MIN_NATIVE_WIDTH = 576
MIN_NATIVE_HEIGHT = 1024

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
        "required_native_width":MIN_NATIVE_WIDTH,
        "required_native_height":MIN_NATIVE_HEIGHT,
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
    """Verify 30 original takes with continuity in the CORRECT hash domain.

    conditioning_input_frame_sha256 and next_conditioning_frame_sha256 are
    SHA-256 of the exact canonical PNG files supplied to the I2V worker.
    last_decoded_frame_sha256 is SHA-256 of raw decoded RGB pixels. Those
    byte domains are intentionally different and MUST NEVER be compared.

    Each non-final take exports its last decoded RGB frame to one canonical
    PNG; the next take must consume that exact PNG (same file SHA). The raw
    decoded RGB digest remains an independent duplicate-tail detector.
    """
    p=plan()
    reasons=[]
    if not isinstance(evidence, Mapping):
        return {"schema": 8, "status": "BLOCKED",
                "reasons": ["INVALID_EVIDENCE_MAPPING"],
                "genuinely_original_segments_verified": 0,
                "expected_original_segments": p["new_true_inference_segment_count"],
                "native_576x1024_detail_verified": False,
                "full_40s_movie_ready": False, "social_published": False}
    if set(evidence)!=set(SEGMENT_COUNTS):
        reasons.append("REQUIRED_7_UNIQUE_SCENE_EVIDENCE_MISSING")
    mp4_seen=set()
    decoded_tails_seen=set()
    conditioning_inputs_seen=set()
    conditioning_exports_seen=set()
    genuine_count=0
    for row in p["scenes"]:
        sid=row["scene_id"]
        receipts=evidence.get(sid,[])
        if not isinstance(receipts,list) or len(receipts)!=row["native_segment_count"]:
            reasons.append(f"{sid}_MISSING_DISTINCT_NATIVE_I2V_SEGMENTS")
            continue
        previous_conditioning_export=None
        chain_valid=True
        for idx,proof in enumerate(receipts):
            if not isinstance(proof,dict):
                reasons.append(f"{sid}_{idx}_NO_GPU_RECEIPT")
                chain_valid=False
                previous_conditioning_export=None
                continue
            mp4_sha=proof.get("mp4_sha256")
            # File SHA of the exact PNG used as I2V conditioning input.
            conditioning_input=proof.get("conditioning_input_frame_sha256")
            # SHA of raw decoded RGB bytes, not a PNG/file SHA.
            decoded_tail=proof.get("last_decoded_frame_sha256")
            # File SHA of the canonical PNG exported from decoded_tail and
            # actually passed to the next segment as its conditioning input.
            next_conditioning=proof.get("next_conditioning_frame_sha256")
            actor_sha=proof.get("canonical_actor_sha256")
            native_width=proof.get("native_width")
            native_height=proof.get("native_height")
            native_detail_ok=(
                type(native_width) is int and native_width>=MIN_NATIVE_WIDTH
                and type(native_height) is int and native_height>=MIN_NATIVE_HEIGHT
            )
            if not native_detail_ok:
                reasons.append(f"{sid}_{idx}_NATIVE_576X1024_DETAIL_NOT_PROVEN")
            if idx < len(receipts)-1:
                next_export_ok=isinstance(next_conditioning,str) and HEX64.fullmatch(next_conditioning)
                if not next_export_ok:
                    reasons.append(f"{sid}_{idx}_NEXT_CONDITIONING_EXPORT_SHA_MISSING")
            else:
                next_export_ok=(next_conditioning is None or
                    (isinstance(next_conditioning,str) and HEX64.fullmatch(next_conditioning)))
                if not next_export_ok:
                    reasons.append(f"{sid}_{idx}_NEXT_CONDITIONING_EXPORT_SHA_INVALID")
            core_valid=(
                all(isinstance(x,str) and HEX64.fullmatch(x)
                    for x in (mp4_sha,conditioning_input,decoded_tail,actor_sha))
                and actor_sha==row["canonical_reference_sha256"]
                and mp4_sha not in mp4_seen
                and type(proof.get("native_frames")) is int
                and proof["native_frames"]==33
                and type(proof.get("native_fps")) is int
                and proof["native_fps"]==24
                and native_detail_ok
                and next_export_ok
                and proof.get("actual_model_inference") is True
                and proof.get("rgb_neon_qc")=="PASS"
                and proof.get("identity_artistic_qc")=="PASS"
            )
            if not core_valid:
                reasons.append(f"{sid}_{idx}_REAL_NATIVE_GPU_OR_QA_NOT_PROVEN")
                chain_valid=False
                previous_conditioning_export=(
                    next_conditioning if isinstance(next_conditioning,str)
                    and HEX64.fullmatch(next_conditioning) else None
                )
                continue
            chain_ok=chain_valid
            if idx==0:
                if proof.get("initial_reference_sha256")!=row["canonical_reference_sha256"]:
                    reasons.append(f"{sid}_ORIGINAL_CHARACTER_SHA_MISMATCH")
                    chain_ok=False
            elif previous_conditioning_export is None or conditioning_input!=previous_conditioning_export:
                reasons.append(f"{sid}_{idx}_CONDITIONING_FRAME_CONTINUITY_BROKEN")
                chain_ok=False
            if conditioning_input in conditioning_inputs_seen:
                reasons.append(f"{sid}_{idx}_REUSED_CONDITIONING_INPUT")
                chain_ok=False
            if decoded_tail in decoded_tails_seen:
                reasons.append(f"{sid}_{idx}_REUSED_DECODED_TAIL")
                chain_ok=False
            if idx < len(receipts)-1:
                if next_conditioning==conditioning_input or next_conditioning in conditioning_exports_seen:
                    reasons.append(f"{sid}_{idx}_REUSED_NEXT_CONDITIONING_EXPORT")
                    chain_ok=False
                conditioning_exports_seen.add(next_conditioning)
                previous_conditioning_export=next_conditioning
            else:
                previous_conditioning_export=None
            conditioning_inputs_seen.add(conditioning_input)
            decoded_tails_seen.add(decoded_tail)
            mp4_seen.add(mp4_sha)
            chain_valid=chain_ok
            if chain_ok:
                genuine_count+=1
    fully_verified=(not reasons and genuine_count==p["new_true_inference_segment_count"])
    return {
        "schema":8,
        "status":"SEGMENT_QA_PASS_OWNER_REVIEW_STILL_REQUIRED" if fully_verified else "BLOCKED",
        "reasons":sorted(set(reasons)),
        "genuinely_original_segments_verified":genuine_count,
        "expected_original_segments":p["new_true_inference_segment_count"],
        "native_576x1024_detail_verified":fully_verified,
        "full_40s_movie_ready":False,
        "social_published":False,
    }


def authorize_v8_scale_up(pilot: Mapping[str, object], *,
                         expected_pilot_sha256: str | None = None) -> dict:
    """Authorize the 30-take batch only after a separate 2–4s native-detail pilot.

    The existing a609d40bc3ee motion A/B is only 33 frames / 1.375s at
    320x576. It is useful for visual review, but cannot satisfy the controlled
    2–4s, >=576x1024 pre-scale pilot requirement. The expected SHA must be
    pinned by the caller and independently echoed by the owner's review
    receipt. Numerical motion and RGB-QA never substitute for acting review.
    This gate never submits GPU work or publishes media.
    """
    if not isinstance(pilot, Mapping):
        return {"schema": 8, "scale_up_authorized": False,
                "blocking_gates": ["INVALID_PILOT_EVIDENCE"],
                "gpu_job_submitted": False, "full_movie_ready": False,
                "public_media_export": False}
    sha = pilot.get("pilot_mp4_sha256")
    reviewed_sha = pilot.get("owner_reviewed_mp4_sha256")
    frames = pilot.get("decoded_frames")
    width = pilot.get("native_width")
    height = pilot.get("native_height")
    fps = pilot.get("native_fps")
    duration = pilot.get("pilot_duration_seconds")
    try:
        duration_value = float(duration)
        duration_ok = (
            isinstance(duration, (int, float)) and not isinstance(duration, bool)
            and math.isfinite(duration_value) and 2.0 <= duration_value <= 4.0
            and type(frames) is int
            and abs(frames / FPS - duration_value) <= 0.5 / FPS
        )
    except (TypeError, ValueError, OverflowError):
        duration_ok = False
    checks = {
        "expected_pilot_sha_not_pinned": (
            isinstance(expected_pilot_sha256, str)
            and HEX64.fullmatch(expected_pilot_sha256) is not None
        ),
        "authentic_pilot_sha_not_owner_reviewed": (
            isinstance(sha, str) and HEX64.fullmatch(sha) is not None
            and sha == expected_pilot_sha256 and reviewed_sha == expected_pilot_sha256
        ),
        "controlled_2_to_4_second_pilot_missing": duration_ok,
        "native_576x1024_detail_missing": (
            type(width) is int and width >= MIN_NATIVE_WIDTH
            and type(height) is int and height >= MIN_NATIVE_HEIGHT
        ),
        "native_24_fps_missing": type(fps) is int and fps == FPS,
        "true_48_to_96_frame_decode_missing": (
            type(frames) is int and 48 <= frames <= 96
        ),
        "true_neural_inference_not_proven": pilot.get("actual_model_inference") is True,
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
