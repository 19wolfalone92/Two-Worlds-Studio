"""Fail-closed EB01 V8 final mux technical verification (private probe metadata only).

Run ffprobe on the PRIVATE final movie; pass its sanitized JSON result here.
This module never accepts, downloads, logs or exports media bytes or URLs.
Technical acceptance is not artistic approval or proof of neural provenance.
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import math

from studio.eb01_v8_40s_native_motion_plan import plan


def verify_final_mux(probe: Mapping, *, decoded_video_frames: int,
                     independent_scene_count: int,
                     canonical_audio_sha_verified: bool,
                     owner_final_artistic_review: bool = False) -> dict:
    reasons = []
    if not isinstance(probe, Mapping):
        return {"status": "BLOCKED", "reasons": ["INVALID_PROBE"], "movie_ready": False}
    streams = probe.get("streams", [])
    if not isinstance(streams, list):
        streams = []
    video = [s for s in streams if isinstance(s, Mapping) and s.get("codec_type") == "video"]
    audio = [s for s in streams if isinstance(s, Mapping) and s.get("codec_type") == "audio"]
    if len(video) != 1:
        reasons.append("EXACTLY_ONE_VIDEO_STREAM_REQUIRED")
    if len(audio) != 1:
        reasons.append("EXACTLY_ONE_AUDIO_STREAM_REQUIRED")
    expected = plan()
    if type(decoded_video_frames) is not int or decoded_video_frames != expected["final_unique_native_frames"]:
        reasons.append("NATIVE_FRAME_COUNT_MISMATCH")
    if type(independent_scene_count) is not int or independent_scene_count != 7:
        reasons.append("SEVEN_SCENES_NOT_PROVEN")
    if canonical_audio_sha_verified is not True:
        reasons.append("SIX_LOCKED_VOICES_NOT_PROVEN")
    if video:
        v = video[0]
        if type(v.get("width")) is not int or v["width"] < 576 or type(v.get("height")) is not int or v["height"] < 1024:
            reasons.append("RESOLUTION_BELOW_TARGET")
        try:
            fps = Fraction(str(v.get("avg_frame_rate", "0/1")))
            r_fps = Fraction(str(v.get("r_frame_rate", "0/1")))
            if fps != 24 or r_fps != 24:
                reasons.append("NOT_NATIVE_24_FPS")
        except (ValueError, ZeroDivisionError):
            reasons.append("INVALID_FPS")
        if v.get("codec_name") != "h264":
            reasons.append("H264_VIDEO_REQUIRED")
    if audio and audio[0].get("codec_name") != "aac":
        reasons.append("AAC_AUDIO_REQUIRED")
    fmt = probe.get("format", {})
    try:
        duration = float(fmt.get("duration", "nan"))
        if not math.isfinite(duration) or duration < 40 or abs(duration - decoded_video_frames / 24) > 0.12:
            reasons.append("INVALID_FINAL_DURATION")
    except (TypeError, ValueError, AttributeError):
        reasons.append("INVALID_FINAL_DURATION")
    if owner_final_artistic_review is not True:
        reasons.append("OWNER_FINAL_ARTISTIC_REVIEW_MISSING")
    return {"status": "TECHNICAL_AND_OWNER_GATES_PASS" if not reasons else "BLOCKED",
            "reasons": sorted(set(reasons)), "movie_ready": not reasons,
            "neural_provenance_independently_proven": False,
            "public_media_export": False}
