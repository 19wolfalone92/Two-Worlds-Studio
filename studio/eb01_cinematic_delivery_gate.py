"""Elizabeth Bones strict cinematic delivery gate.

Three separate facts:
  (1) video/audio files can be decoded;
  (2) independent neural performance, voice-conditioned lip sync and
      identity continuity are independently evidenced;
  (3) the owner has *personally* approved the artistic quality.

A passed FFmpeg job proves only (1). It must never silently establish (2)
or (3). No frame synthesis, media regeneration, paid service, or publishing.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCENES = ("S00", "S01", "S02", "S03", "S04", "S05", "S06")
DIALOGUE = ("S01", "S02", "S03", "S04", "S05", "S06")
# A human face needs cheek/jaw lip-sync; a skeleton has *no cheeks*.
# A generic human-face animation model could add skin to Kostya, violating
# the original character identity despite good audio correlation.
ACTOR_BY_SCENE = {
    "S00":"KOSTYA","S01":"ELIZABETH","S02":"KOSTYA",
    "S03":"ELIZABETH","S04":"KOSTYA","S05":"ELIZABETH",
    "S06":"KOSTYA",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MIN_SECONDS = 40.0
MIN_WIDTH = 576
MIN_HEIGHT = 1024
MIN_FPS = 24


def _positive_int(value: Any) -> int:
    return value if type(value) is int and value > 0 else 0


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _no_reuse(evidence: dict, sid: str) -> bool:
    """Reject fake-new shots composed of footage from another scene."""
    if evidence.get("original_native_ai_inference") is not True:
        return False
    if evidence.get("reused_from_other_scene") is True:
        return False
    if evidence.get("reused_frames_from_S00_S04_S06") is True:
        return False
    if evidence.get("reuses_other_scene_frames") is True:
        return False
    if evidence.get("frames_reused_elsewhere_in_film") is True:
        return False
    segments = evidence.get("source_segments")
    if isinstance(segments, list) and any(
        isinstance(s, dict) and (s.get("reused_from_another_scene") is True
        or s.get("new_original_S02_inference") is False)
        for s in segments
    ):
        return False
    return True


def _is_proven_actor(scene: dict, sid: str) -> bool:
    """No self-signed synthetic clip can pass an independent-source test."""
    sha = scene.get("source_sha256", scene.get("sha256"))
    return (
        _no_reuse(scene, sid)
        and _valid_digest(sha)
        and scene.get("real_neural_video_verified") is True
        and _valid_digest(scene.get("model_receipt_sha256"))
        and _valid_digest(scene.get("canonical_character_reference_sha256"))
        and scene.get("same_character_identity_qc") == "PASS"
        and scene.get("physically_coherent_movement_qc") == "PASS"
        and scene.get("human_visual_frame_review") == "PASS"
        and scene.get("neon_color_qc") == "PASS"
    )


def assess(
    report: dict,
    *,
    evidence: dict[str, dict] | None = None,
    owner_artistic_approval: bool = False,
) -> dict:
    """Classify based on proof, never just on a CI pipeline result.

    evidence contains *independently checked* per-scene QA. Do not derive
    PASS from the model's own receipt. A human approval is a separate input.
    """
    if not isinstance(report, dict):
        raise ValueError("JSON object with actual film probe and QA is required")
    if evidence is None:
        evidence = report.get("independent_scene_evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("independent_scene_evidence must be a dictionary")
    reasons = []

    def require(ok: bool, why: str) -> None:
        if not ok:
            reasons.append(why)

    seconds = report.get("duration_seconds")
    require(type(seconds) in (int, float) and MIN_SECONDS <= seconds < 300,
            "DURATION_SHORTER_THAN_40_SECONDS")
    resolution = report.get("resolution")
    if not isinstance(resolution, (tuple, list)):
        resolution = [report.get("width"), report.get("height")]
    require(len(resolution) == 2
            and _positive_int(resolution[0]) >= MIN_WIDTH
            and _positive_int(resolution[1]) >= MIN_HEIGHT,
            "INSUFFICIENT_NATIVE_VERTICAL_RESOLUTION")
    fps = report.get("fps")
    require(_positive_int(fps) >= MIN_FPS, "FPS_BELOW_24")
    require(report.get("russian_voice_lines") == 6,
            "MISSING_SIX_ORIGINAL_RUSSIAN_VOICE_LINES")
    require(report.get("russian_subtitle_cues") == 6,
            "MISSING_SIX_RUSSIAN_SUBTITLE_CUES")
    require(report.get("scenes_on_timeline") == 7,
            "NOT_SEVEN_INDEPENDENT_SCENES")
    require(report.get("original_artistic_quality_approved") is not False
            or not owner_artistic_approval,
            "CONFLICTING_UNAPPROVED_OWNER_STATUS")

    # A scan may detect suspicious frames; a negative scan is NOT proof of
    # photoreal acting quality or facial/wardrobe identity.
    suspect = report.get("suspected_colored_neon_frames")
    require(_positive_int(suspect) == 0 and (suspect is None or type(suspect) is int),
            "NEON_COLOR_SPECTRAL_ARTIFACTS")
    require(report.get("full_visual_frame_qa_status") == "PASS",
            "INDEPENDENT_FULL_FRAME_VISUAL_QA_MISSING")

    require(set(evidence) == set(SCENES), "MISSING_INDEPENDENT_SCENE_RECEIPTS")
    unique_sources = set()
    for sid in SCENES:
        row = evidence.get(sid, {})
        if not isinstance(row, dict):
            row = {}
        require(_no_reuse(row, sid), f"{sid}_FOOTAGE_REUSE_OR_NO_ORIGINAL_INFERENCE")
        require(row.get("actor_id") == ACTOR_BY_SCENE[sid],
                f"{sid}_ACTOR_IDENTITY_PROVENANCE_MISSING")
        require(_is_proven_actor(row, sid), f"{sid}_ORIGINAL_VIDEO_IDENTITY_MOTION_UNVERIFIED")
        if ACTOR_BY_SCENE[sid] == "KOSTYA":
            require(row.get("skeleton_anatomy_preserved") == "PASS",
                    f"{sid}_BONE_ANATOMY_OR_SUIT_DRIFT_UNVERIFIED")
        media_sha = row.get("source_sha256", row.get("sha256"))
        if _valid_digest(media_sha):
            require(media_sha not in unique_sources, f"{sid}_DUPLICATE_SOURCE_CLIP")
            unique_sources.add(media_sha)
        if sid in DIALOGUE:
            require(_valid_digest(row.get("original_qwen_voice_sha256")),
                    f"{sid}_VOICE_ORIGINAL_NOT_LOCKED")
            human_expression = row.get("mouth_jaw_cheek_motion_review") == "PASS"
            bone_expression = row.get("bone_jaw_hinge_motion_review") == "PASS"
            appropriate_expression = (human_expression if ACTOR_BY_SCENE[sid] == "ELIZABETH"
                                      else bone_expression)
            require(row.get("audio_conditioned_lipsync") == "PASS"
                    and appropriate_expression,
                    f"{sid}_NO_VERIFIED_AUDIO_DRIVEN_LIPSYNC")
    # Owner approval cannot override actual broken content/provenance.
    if reasons:
        status = "BLOCKED"
    elif owner_artistic_approval is True:
        status = "RELEASE_REVIEW_PASS"
    else:
        status = "OWNER_REVIEW_REQUIRED"
    return {
        "schema": 1,
        "status": status,
        "cinematic_delivery_approved": status == "RELEASE_REVIEW_PASS",
        "owner_review_still_required": status != "RELEASE_REVIEW_PASS",
        "independent_performance_scenes_proven": len(SCENES) if not reasons else 0,
        "reasons": sorted(set(reasons)),
        "source_film_sha256": report.get("sha256"),
        "published": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Fail closed on counterfeit or unreviewed cinema")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--evidence", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    raw = json.loads(a.report.read_text(encoding="utf-8"))
    evidence = json.loads(a.evidence.read_text(encoding="utf-8")) if a.evidence else None
    result = assess(raw, evidence=evidence)
    message = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    print(message)
    if a.output:
        if a.output.exists():
            raise FileExistsError("Never silently overwrite prior release QA")
        a.output.write_text(message, encoding="utf-8")
    return 0 if result["status"] == "OWNER_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
