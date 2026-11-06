"""Story Glitch cinematic continuity ledger for real shot manifests.

This checks *declared* identity, wardrobe, geometry and prop anchors. It is
not image similarity, facial recognition or an aesthetic certification.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HASH = re.compile(r"^[0-9a-f]{64}$")
ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{1,80}$")


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError(f"Invalid {label} continuity identifier")
    return value


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError(f"Missing/invalid {label} SHA-256 anchor")
    return value


def audit(shots: list[dict], *, strict: bool = False) -> dict:
    """All shots in global chronological order; immutable reference anchor IDs.

    Optional approved scene transitions are explicit, never inferred from
    neighboring prompts, filenames, or neural face resemblance.
    """
    if not isinstance(shots, list) or not 1 <= len(shots) <= 240:
        raise ValueError("Expected a nonempty ordered shot list")
    identities, wardrobe, geometries, objects, variants = {}, {}, {}, {}, {}
    missing = 0
    entries = []
    for position, row in enumerate(shots, 1):
        if not isinstance(row, dict):
            raise ValueError("Shot continuity row must be an object")
        scene = row.get("scene_id", f"S{position:03d}")
        c = row.get("continuity")
        if c is None:
            missing += 1
            entries.append({"scene": scene, "status": "unverifiable_no_metadata"})
            continue
        if not isinstance(c, dict):
            raise ValueError(f"{scene}: continuity metadata must be an object")
        location = c.get("location")
        cast = c.get("cast", [])
        props = c.get("props", [])
        transitions = c.get("transitions", [])
        if not isinstance(cast, list) or not isinstance(props, list) or not isinstance(transitions, list):
            raise ValueError(f"{scene}: cast, props and transitions must be arrays")
        if len(cast) > 8 or len(props) > 25 or len(transitions) > 12:
            raise ValueError(f"{scene}: excessive continuity metadata")
        permitted = set()
        for change in transitions:
            if not isinstance(change, dict):
                raise ValueError(f"{scene}: invalid continuity transition")
            kind = change.get("kind")
            subject = _identifier(change.get("id"), "transition")
            if kind not in {"wardrobe", "age_variant"}:
                raise ValueError(f"{scene}: invalid continuity transition kind")
            reason = str(change.get("reason", "")).strip()
            reviewer = str(change.get("reviewed_by", "")).strip()
            if len(reason) < 12 or (strict and len(reviewer) < 3):
                raise ValueError(f"{scene}: continuity change requires reason and reviewer")
            permitted.add((kind, subject))
        if location is not None:
            if not isinstance(location, dict):
                raise ValueError(f"{scene}: location must be a dictionary")
            lid = _identifier(location.get("id"), "location")
            layout = _hash(location.get("layout_sha256"), "location")
            if lid in geometries and geometries[lid] != layout:
                raise ValueError(f"{scene}: location {lid} layout geometry drift")
            geometries[lid] = layout
        for role in cast:
            if not isinstance(role, dict):
                raise ValueError(f"{scene}: cast member must be an object")
            body = _identifier(role.get("body_id"), "body")
            version = _identifier(role.get("age_variant"), "age variant")
            ref = _hash(role.get("reference_sha256"), "actor reference")
            outfit = _identifier(role.get("wardrobe_id"), "wardrobe")
            anchor = (body, version)
            if anchor in identities and identities[anchor] != ref:
                raise ValueError(f"{scene}: {body} visual identity reference changed")
            if body in variants and variants[body] != version:
                if ("age_variant", body) not in permitted:
                    raise ValueError(f"{scene}: {body} age variant changed without narrative transition")
            if body in wardrobe and wardrobe[body] != outfit:
                if ("wardrobe", body) not in permitted:
                    raise ValueError(f"{scene}: {body} costume changed without narrative transition")
            identities[anchor], variants[body], wardrobe[body] = ref, version, outfit
        for prop in props:
            if not isinstance(prop, dict):
                raise ValueError(f"{scene}: prop must be a dictionary")
            pid = _identifier(prop.get("id"), "prop")
            ref = _hash(prop.get("reference_sha256"), "prop")
            if pid in objects and objects[pid] != ref:
                raise ValueError(f"{scene}: prop {pid} reference changed across frames")
            objects[pid] = ref
        entries.append({"scene": scene, "status": "structural_reference_consistent",
                        "cast_count": len(cast), "props_count": len(props),
                        "approved_transition_metadata": bool(transitions)})
    if strict and missing:
        raise ValueError(f"Strict master has {missing} shots with no continuity metadata")
    return {
        "shots": len(shots), "metadata_coverage": len(shots) - missing,
        "unverifiable_shots": missing,
        "character_anchors": len(identities), "location_anchors": len(geometries),
        "prop_anchors": len(objects),
        "structural_continuity_pass": missing == 0,
        "actual_face_similarity_verified": False,
        "actual_location_appearance_verified": False,
        "human_reference_comparison_required": True,
        "shot_reports": entries,
    }


def main():
    p = argparse.ArgumentParser(description="Metadata-only continuity of fictional visual references")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    shots = document.get("shots")
    print(json.dumps(audit(shots, strict=args.strict), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
