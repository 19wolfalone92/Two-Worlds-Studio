"""Strict core manifest contracts, with extensible, versioned Work scene fields.

Pydantic is an optional preflight dependency. This module never contacts a
provider, edits source episodes, approves artwork, or publishes content.
Call validate_document() before any private production request.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StudioModel(BaseModel):
    # Allow forward-compatible Work scene extensions, but validate typed core.
    model_config = ConfigDict(extra="allow", strict=True)


class Scene(StudioModel):
    id: str = Field(pattern=r"^S[0-9]{2,4}$")
    duration: float | int = Field(gt=0, le=300)
    shot: str = Field(min_length=10, max_length=4000)
    palette: str | None = None
    subjects: list[str] = Field(default_factory=list)
    continuity: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_subjects(self):
        if len(self.subjects) != len(set(self.subjects)):
            raise ValueError("Duplicate cast subject in one scene")
        if any(not isinstance(subject, str) or not subject.strip() for subject in self.subjects):
            raise ValueError("Invalid cast subject")
        return self


class Episode(StudioModel):
    id: str = Field(min_length=2, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    title: str = Field(min_length=1, max_length=180)
    fps: int = Field(default=30, ge=1, le=120)
    scenes: list[Scene] = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def check_scenes(self):
        ids = [scene.id for scene in self.scenes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate scene ID")
        return self


class VisualCharacter(StudioModel):
    identity_id: str = Field(min_length=1)
    body_name: str = Field(min_length=1)
    gender: str = Field(min_length=1)
    age: int = Field(ge=0, le=125)
    body: str = Field(min_length=1)
    distinctive: str = Field(min_length=1)
    clothing: str = Field(min_length=1)
    seed_anchor: int = Field(ge=0)
    reference_file: str | None = None


class VisualBible(StudioModel):
    schema: Literal[1]
    visual_style: str = Field(min_length=10)
    composition: str = Field(min_length=10)
    negative: str = Field(min_length=1)
    locations: dict[str, str] = Field(min_length=1)
    characters: dict[str, VisualCharacter] = Field(min_length=1)


class IdentityReference(StudioModel):
    view: Literal["front", "three_quarter", "profile", "full_body", "expression"]
    file: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_status: Literal["approved", "unreviewed"] = "unreviewed"


class IdentityBody(StudioModel):
    version: str = Field(min_length=1)
    references: list[IdentityReference] = Field(default_factory=list)


class IdentityRegistry(StudioModel):
    schema: Literal[1]
    bodies: dict[str, IdentityBody] = Field(min_length=1)


MODELS = {"episode": Episode, "visual_bible": VisualBible, "identities": IdentityRegistry}


def validate_document(path: Path, kind: str) -> dict:
    """Fail early on bad JSON/schema. Print only status/count, never private story text."""
    if kind not in MODELS:
        raise ValueError("Unknown contract kind")
    path = Path(path)
    if path.stat().st_size > 2_000_000:
        raise ValueError("JSON contract exceeds 2 MB")
    data = json.loads(path.read_text(encoding="utf-8"))
    model = MODELS[kind].model_validate(data)
    count = (
        len(model.scenes) if kind == "episode"
        else len(model.characters) if kind == "visual_bible"
        else len(model.bodies)
    )
    return {"kind": kind, "status": "validated", "entries": count,
            "schema_validation": "pydantic_v2", "source_edited": False,
            "review_approved": False, "network_used": False}


def validate_bundle(episode: dict, bible: dict, identities: dict) -> dict:
    """Check declared scene locations/physical bodies against locked registries."""
    ep = Episode.model_validate(episode)
    visual = VisualBible.model_validate(bible)
    registry = IdentityRegistry.model_validate(identities)
    missing = set(visual.characters) - set(registry.bodies)
    if missing:
        raise ValueError("Visual bible has body IDs absent from identity registry")
    for scene in ep.scenes:
        if scene.palette is not None and scene.palette not in visual.locations:
            raise ValueError("Episode refers to unknown visual location")
        refs = set(scene.subjects)
        body = scene.continuity.get("body")
        if body is not None:
            if not isinstance(body, str):
                raise ValueError("Physical body ID must be text")
            refs.add(body)
        if refs - set(visual.characters):
            raise ValueError("Episode refers to unknown physical body")
    return {"status": "validated", "scenes": len(ep.scenes),
            "bodies": len(registry.bodies), "review_approved": False}
