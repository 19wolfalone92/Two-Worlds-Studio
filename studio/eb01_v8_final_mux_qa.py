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
                     owner_final_artistic_review: bool = False,
                     original_neural_segments_verified: bool = False,
                     six_line_lipsync_verified: bool = False,
                     private_durable_storage_verified: bool = False,
                     native_576x1024_detail_verified: bool = False) -> dict:
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
    # Independently inspect the encoded video stream, not only caller-provided
    # decoded frame counts or container duration. A valid-looking container
    # can contain a truncated stream or a stale/incorrect frame count.
    if video:
        v = video[0]
        try:
            n = int(v.get("nb_frames", ""))
            if str(v.get("nb_frames")) != str(n) or n != expected["final_unique_native_frames"]:
                reasons.append("ENCODED_VIDEO_FRAME_COUNT_MISMATCH")
        except (TypeError, ValueError, OverflowError):
            reasons.append("ENCODED_VIDEO_FRAME_COUNT_MISSING")
        try:
            stream_duration = float(v.get("duration"))
            if not math.isfinite(stream_duration) or abs(stream_duration - expected["final_duration_seconds"]) > 0.12:
                reasons.append("ENCODED_VIDEO_DURATION_MISMATCH")
        except (TypeError, ValueError, OverflowError):
            reasons.append("ENCODED_VIDEO_DURATION_MISSING")
    if audio:
        try:
            audio_duration = float(audio[0].get("duration"))
            if not math.isfinite(audio_duration) or abs(audio_duration - expected["final_duration_seconds"]) > 0.12:
                reasons.append("AUDIO_STREAM_DURATION_MISMATCH")
        except (TypeError, ValueError, OverflowError):
            reasons.append("AUDIO_STREAM_DURATION_MISSING")
    if audio and audio[0].get("codec_name") != "aac":
        reasons.append("AAC_AUDIO_REQUIRED")
    fmt = probe.get("format", {})
    try:
        if not isinstance(fmt, Mapping) or type(decoded_video_frames) is not int:
            raise ValueError("Invalid format or decoded frame count")
        duration = float(fmt.get("duration", "nan"))
        if not math.isfinite(duration) or duration < 40 or abs(duration - decoded_video_frames / 24) > 0.12:
            reasons.append("INVALID_FINAL_DURATION")
    except (TypeError, ValueError, AttributeError, OverflowError):
        reasons.append("INVALID_FINAL_DURATION")
    if original_neural_segments_verified is not True:
        reasons.append("ORIGINAL_NEURAL_PROVENANCE_MISSING")
    if native_576x1024_detail_verified is not True:
        # Container dimensions can be obtained by upscaling 320x576 footage.
        # That is not native model detail and must never satisfy the cinematic
        # resolution requirement by itself.
        reasons.append("NATIVE_576X1024_DETAIL_UNVERIFIED")
    if six_line_lipsync_verified is not True:
        reasons.append("SIX_LINE_LIPSYNC_MISSING")
    if private_durable_storage_verified is not True:
        reasons.append("PRIVATE_DURABLE_STORAGE_UNVERIFIED")
    if owner_final_artistic_review is not True:
        reasons.append("OWNER_FINAL_ARTISTIC_REVIEW_MISSING")
    return {
        "status": "TECHNICAL_AND_OWNER_GATES_PASS" if not reasons else "BLOCKED",
        "reasons": sorted(set(reasons)),
        "movie_ready": not reasons,
        "neural_provenance_independently_proven": original_neural_segments_verified is True,
        "native_576x1024_detail_independently_proven": native_576x1024_detail_verified is True,
        "public_media_export": False,
    }
