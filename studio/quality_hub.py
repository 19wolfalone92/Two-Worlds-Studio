"""Single read-only release QC summary across existing technical validators.

Never claims visual/acting/rights approval from offline tests. Does not write
source episodes or publish; optional ASR input requires a real transcript.
"""
from __future__ import annotations

import json
from pathlib import Path


def assess(video: Path, srt: Path, cover: Path, *,
           episode: Path | None = None, assets: Path | None = None,
           original_transcript: Path | None = None,
           recognized_transcript: Path | None = None,
           full_decode: bool = False, master: bool = False,
           language: str = "ru") -> dict:
    """Run fail-closed checks; report safe categories without leaking private media."""
    if language not in {"ru", "en"}:
        raise ValueError("Only ru/en language settings are supported")
    if (original_transcript is None) != (recognized_transcript is None):
        raise ValueError("Voice comparison requires BOTH real source and recognized transcripts")
    if master and episode is None:
        raise ValueError("Master check requires the source episode for identity review")
    checks: dict[str, dict] = {}
    blocked = False
    human_review = True  # no software-only release approval

    def run_check(name, fn):
        nonlocal blocked
        try:
            checks[name] = {"status": "technical_pass", "details": fn()}
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
            blocked = True
            # Never publish untrusted private file paths, content or remote errors.
            checks[name] = {"status": "blocked", "error_type": type(exc).__name__}

    from studio.release_gate import check_video
    run_check("video_subtitles_cover", lambda: check_video(
        Path(video), Path(srt), Path(cover),
        require_russian=language == "ru", full_decode=full_decode,
    ))

    if episode is not None:
        from studio.identity_qc import verify_episode_characters
        def verify_identities():
            path = Path(episode)
            source = json.loads(path.read_text(encoding="utf-8"))
            return verify_episode_characters(
                source, Path(assets) if assets else path.parent / "assets",
                strict=master,
            )
        run_check("physical_character_identity", verify_identities)

    if original_transcript is not None:
        from studio.voice_audit import audit_transcript
        def verify_words():
            nonlocal human_review
            report = audit_transcript(
                Path(original_transcript).read_text(encoding="utf-8"),
                Path(recognized_transcript).read_text(encoding="utf-8"),
            )
            if report["requires_review"]:
                human_review = True
            # Never print source sentences, observed sentences or individual edits.
            return {"word_error_rate": report["word_error_rate"],
                    "critical_differences": len(report["critical_differences"]),
                    "human_listening_performed": False}
        run_check("independent_transcript_comparison", verify_words)

    return {
        "schema": 1,
        "status": "blocked" if blocked else "technical_pass_human_review_required",
        "checks": checks,
        "human_review_required": human_review,
        "rights_approved": False,
        "visual_continuity_approved": False,
        "voice_acting_approved": False,
        "publication_authorized": False,
        "published": False,
    }
