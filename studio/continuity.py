"""Executable canonical continuity checks; tracking consciousness *and* body.

The story editor still reviews language, motivations and chronology; this checker
catches core identity contradictions before an episode is rendered.
"""
from __future__ import annotations

BODIES = {
    "earth_mira": {"pre": "Mira", "post": "Lyra"},
    "noctaris_lyra": {"pre": "Lyra", "post": "Mira"},
    "earth_eli": {"pre": "Ilan", "post": "Ilan"},
    "noctaris_ilan": {"pre": "Eli", "post": "Eli"},
}
STATES = {"pre", "transition", "post"}


def check_episode(episode: dict, *, strict: bool = False) -> dict:
    """Validate timeline of the Mira/Lyra swap, plus all centrally depicted bodies.

    For samples without continuity metadata this returns checked=False unless
    strict=True. Approved schedule inputs must use strict mode.
    """
    has_state = "continuity_start" in episode or "continuity_end" in episode
    if not has_state:
        if strict:
            raise ValueError("Cannot stage an approved episode without continuity metadata")
        return {"checked": False, "reason": "continuity metadata not provided"}
    start = episode.get("continuity_start")
    end = episode.get("continuity_end")
    if start not in ("pre", "post") or end not in ("pre", "post"):
        raise ValueError("Invalid continuity start/end state")
    state = start
    transitions = 0
    scenes = episode.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Cannot validate an empty episode")
    for scene in scenes:
        ref = scene.get("continuity")
        if not isinstance(ref, dict):
            raise ValueError(f"{scene.get('id')}: missing continuity annotation")
        stage = ref.get("state")
        if stage not in STATES:
            raise ValueError(f"{scene.get('id')}: invalid continuity state")
        if stage == "transition":
            if state != "pre":
                raise ValueError("Exchange transition can only occur in the pre state")
            if ref.get("event") != "mira_lyra_exchange":
                raise ValueError("Unrecognized mirror exchange transition")
            if ref.get("body") is not None or ref.get("occupant") is not None:
                raise ValueError("Transition scene must not assign an occupant")
            state = "post"
            transitions += 1
            continue
        if stage != state:
            raise ValueError(f"{scene.get('id')}: chronological state regression")
        body = ref.get("body")
        occupant = ref.get("occupant")
        if body is not None:
            if body not in BODIES:
                raise ValueError(f"{scene.get('id')}: unknown body {body}")
            expected = BODIES[body][state]
            if occupant != expected:
                raise ValueError(
                    f"{scene.get('id')}: {body} belongs to consciousness "
                    f"{expected} after state={state}, not {occupant}"
                )
        elif occupant is not None:
            raise ValueError("An occupant without a body cannot be validated")
    if state != end:
        raise ValueError(f"Episode end state {end} mismatches computed state {state}")
    return {"checked": True, "start": start, "end": end,
            "exchange_count": transitions, "scenes_checked": len(scenes)}


def check_sequence(episodes: list[dict]) -> dict:
    """Check contiguous serialized episodes without magically undoing a swap."""
    if not episodes:
        raise ValueError("No episodes supplied")
    prev_end = None
    reviews = []
    for ep in episodes:
        result = check_episode(ep, strict=True)
        if prev_end is not None and result["start"] != prev_end:
            raise ValueError(f"{ep.get('id')}: continuity break between episodes")
        prev_end = result["end"]
        reviews.append({"episode":ep.get("id"), **result})
    return {"checked": True, "episodes": reviews}
