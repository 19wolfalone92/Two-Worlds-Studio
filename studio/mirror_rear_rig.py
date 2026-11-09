"""Mirror staging contract: Mira's back faces the camera, Lyra is beyond the glass.

This module deliberately refuses to animate the Work front-facing portrait as
a back view. A rear-view multipart rig must be authored and kept private.

The metadata is a release-blocking contract, not computer vision proof of
correct anatomy or character identity. Visual owner review remains mandatory.
"""
from __future__ import annotations

import json
from pathlib import Path

from studio.character_engine import CharacterRig

REQUIRED_PARTS = frozenset({
    "torso", "head", "upper_arm", "forearm", "hand",
    "left_leg", "right_leg",
})
REQUIRED_MOTION = ("__root__", "head", "upper_arm", "forearm",
                   "left_leg", "right_leg")
ANIMATED_FIELDS = ("x", "y", "angle", "scale")


def _moves(keys: object) -> bool:
    if not isinstance(keys, list) or len(keys) < 2:
        return False
    first = keys[0]
    if not isinstance(first, dict):
        return False
    return any(
        isinstance(key, dict) and any(
            abs(float(key.get(field, 1 if field == "scale" else 0)) -
                float(first.get(field, 1 if field == "scale" else 0))) > 1e-5
            for field in ANIMATED_FIELDS
        )
        for key in keys[1:]
    )


def validate_mira_rear_contract(config: dict) -> None:
    """Reject the front-facing cutout and sticker-like root-only movement.

    This is a structural check. A human must *also* verify the actual view.
    """
    if not isinstance(config, dict):
        raise ValueError("Mira rear rig must be a JSON object")
    if (config.get("body_id") != "earth_mira"
            or config.get("camera_view") != "back"
            or config.get("facing") != "mirror"):
        raise ValueError(
            "Mira must face the mirror with her BACK to the camera "
            "(body_id=earth_mira, camera_view=back, facing=mirror)"
        )
    if config.get("schema") != 1:
        raise ValueError("Use the Studio Character Animation Engine schema=1")
    parts = config.get("parts", [])
    names = {p.get("name") for p in parts if isinstance(p, dict)}
    missing = REQUIRED_PARTS - names
    if missing:
        raise ValueError("Rear-view articulated rig is missing: " +
                         ", ".join(sorted(missing)))
    for part in parts:
        if not isinstance(part, dict) or not isinstance(part.get("sha256"), str) or len(part["sha256"]) != 64:
            raise ValueError("Every rear-view rig sprite requires pinned SHA-256")
    tracks = config.get("tracks", {})
    if not isinstance(tracks, dict):
        raise ValueError("Rear-view rig needs animated motion tracks")
    missing_motion = [name for name in REQUIRED_MOTION
                      if not _moves(tracks.get(name))]
    if missing_motion:
        raise ValueError("Rear-view shot requires articulated motion for: " +
                         ", ".join(missing_motion))
    if not isinstance(config.get("ground_shadow"), dict):
        raise ValueError("Rear-view shot requires a grounded contact shadow")


def load_mira_rear_rig(manifest_path: Path, asset_root: Path,
                       canvas_size: tuple[int, int]) -> CharacterRig:
    """Load only private, explicitly authored BACK-view art and motion.

    No front-view fallback, no horizontal flip, no auto-invented rear view.
    """
    manifest_path = Path(manifest_path)
    asset_root = Path(asset_root)
    config = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_mira_rear_contract(config)
    return CharacterRig(config, asset_root, canvas_size, publish=False)
