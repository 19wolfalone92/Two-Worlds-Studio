"""Load an offline facial performance onto its exact portrait and shot clock."""
from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

from PIL import Image

from studio.premium_assets import LICENSES, resolve_asset


def _manifest(root: Path, spec: dict) -> tuple[Path, dict]:
    if not isinstance(spec, dict) or set(spec) - {"path", "sha256", "review_status", "rights", "rights_note"}:
        raise ValueError("Invalid facial take reference")
    name = spec.get("path")
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Facial take needs a relative JSON path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path.suffix.lower() != ".json":
        raise ValueError("Facial take must be a local JSON file")
    if not path.is_file() or path.stat().st_size > 512 * 1024:
        raise ValueError("Facial take JSON is missing or too large")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec.get("sha256"):
        raise ValueError("Facial take manifest SHA-256 mismatch")
    data = json.loads(raw)
    if (not isinstance(data, dict) or type(data.get("schema")) is not int or data["schema"] != 1
            or not isinstance(data.get("frames"), list) or not 1 <= len(data["frames"]) <= 1800):
        raise ValueError("Invalid facial take manifest")
    return path, data


def _frames(root: Path, data: dict) -> list[tuple[Path, str] | None]:
    result = []
    for frame in data["frames"]:
        if frame is None:
            result.append(None)
            continue
        if (not isinstance(frame, dict) or set(frame) != {"path", "sha256"}
                or not isinstance(frame["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", frame["sha256"])):
            raise ValueError("Facial take frame requires a path and SHA-256")
        path = resolve_asset(root, frame["path"])
        if path.suffix.lower() != ".png":
            raise ValueError("Facial take frames must be RGBA PNGs")
        result.append((path, frame["sha256"]))
    if not any(result):
        raise ValueError("Facial take must include an animated patch")
    return result


def take_sources(root: Path, spec: dict) -> set[Path]:
    manifest, data = _manifest(root, spec)
    return {manifest, *(frame[0] for frame in _frames(root, data) if frame is not None)}


class PerformanceTake:
    """Frame-locked patches are composited in head space before the rig moves."""

    def __init__(self, root: Path, spec: dict, *, canvas: tuple[int, int],
                 head: Image.Image, head_box: tuple[int, int, int, int], fps: int,
                 frames: int, portrait_sha: str, head_sha: str, speech: dict | None,
                 master: bool = False):
        _, data = _manifest(root, spec)
        if master and (spec.get("review_status") != "approved" or spec.get("rights") not in LICENSES
                       or not isinstance(spec.get("rights_note"), str) or len(spec["rights_note"].strip()) < 8):
            raise ValueError("Master facial take needs review and documented rights")
        if (type(data.get("fps")) is not int or data["fps"] != fps or len(data["frames"]) != frames
                or data.get("canvas") != list(canvas)):
            raise ValueError("Facial take does not match the shot frame clock and portrait canvas")
        if data.get("portrait_sha256") != portrait_sha or data.get("head_sha256") != head_sha:
            raise ValueError("Facial take does not match the locked portrait and head texture")
        if "speech" not in data or data["speech"] != speech:
            raise ValueError("Facial take does not match the shot's WAV, text and voice offset")
        box = data.get("box")
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                or not (0 <= box[0] < box[2] <= canvas[0] and 0 <= box[1] < box[3] <= canvas[1])
                or (box[2] - box[0]) * (box[3] - box[1]) > 1_200_000):
            raise ValueError("Invalid facial take patch bounds")
        x0, y0 = max(box[0], head_box[0]), max(box[1], head_box[1])
        x1, y1 = min(box[2], head_box[2]), min(box[3], head_box[3])
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Facial take patch misses the head texture")
        self.fps, self.head = fps, head
        self.head_alpha = head.getchannel("A")
        self.destination = x0 - head_box[0], y0 - head_box[1]
        self.crop = x0 - box[0], y0 - box[1], x1 - box[0], y1 - box[1]
        self.size = box[2] - box[0], box[3] - box[1]
        self.frames = _frames(root, data)
        self.cache_key = None
        self.cache_image = head

    def frame(self, seconds: float) -> Image.Image:
        index = min(len(self.frames) - 1, max(0, int(seconds * self.fps + .5)))
        entry = self.frames[index]
        if entry is None:
            return self.head
        if entry != self.cache_key:
            path, expected = entry
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("Facial take frame SHA-256 mismatch")
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or image.mode != "RGBA" or image.size != self.size:
                    raise ValueError("Facial take frame does not match its RGBA patch dimensions")
                patch = image.crop(self.crop)
            painted = self.head.copy()
            painted.alpha_composite(patch, dest=self.destination)
            painted.putalpha(self.head_alpha)
            self.cache_key, self.cache_image = entry, painted
        return self.cache_image
