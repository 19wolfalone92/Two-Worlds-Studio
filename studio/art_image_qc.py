"""Offline decoded-image technical QA for premium story artwork.

This is NOT subjective composition judging or face/identity certification.
Reports blank/near-black/near-white, focus and tonal-dynamic-range warnings,
plus a lightweight duplicate fingerprint. Never uploads media or phones home.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 8_000_000


class InvalidArtwork(ValueError):
    """Invalid/corrupt/non-image payload, not an aesthetic judgment."""


def inspect_image_bytes(data: bytes, *, expected_size: tuple[int,int] | None = None,
                        require_png: bool = True) -> dict:
    if not isinstance(data, bytes) or not 32 < len(data) <= MAX_IMAGE_BYTES:
        raise InvalidArtwork("Empty or oversized artwork bytes")
    try:
        with Image.open(io.BytesIO(data)) as image:
            actual_format = image.format
            size = image.size
            if size[0] * size[1] > MAX_PIXELS or min(size) < 32:
                raise InvalidArtwork("Artwork pixel count is unsafe or too small")
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()  # Full pixel decode, not only a PNG signature.
            if require_png and image.format != "PNG":
                raise InvalidArtwork("Artwork is not a PNG")
            if expected_size is not None and image.size != expected_size:
                raise InvalidArtwork("Artwork dimensions differ from approved job")
            small = ImageOps.exif_transpose(image).convert("RGB")
            small.thumbnail((256, 448), Image.Resampling.BOX)
            pixels = np.asarray(small, dtype=np.float32)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidArtwork("Artwork cannot be fully decoded") from None
    # Relative sRGB luma is sufficient for diagnostics; it is not a color grade.
    luminance = pixels[:,:,0]*0.2126 + pixels[:,:,1]*0.7152 + pixels[:,:,2]*0.0722
    dark_fraction = float(np.mean(luminance < 12))
    white_fraction = float(np.mean(luminance > 246))
    mean = float(np.mean(luminance))
    deviation = float(np.std(luminance))
    # 4-neighbor Laplacian catches exceptionally blurred outputs.
    # Low values are WARNINGS: intentional soft-focus cinema may be valid.
    center = luminance[1:-1,1:-1]
    lap = 4*center-luminance[:-2,1:-1]-luminance[2:,1:-1] - luminance[1:-1,:-2] - luminance[1:-1,2:]
    detail = float(np.std(lap))
    # Difference hash: useful for detecting suspicious near-identical shots.
    small_hash = Image.fromarray(np.uint8(luminance), "L").resize(
        (9,8), Image.Resampling.BOX)
    samples = np.asarray(small_hash, dtype=np.uint8)
    fingerprint = "".join("1" if x else "0" for x in (samples[:,1:] > samples[:,:-1]).ravel())
    dhash = f"{int(fingerprint,2):016x}"
    issues=[]
    if deviation < 2.0:
        issues.append("uniform_or_blank_image")
    if dark_fraction > .975:
        issues.append("nearly_all_black")
    if white_fraction > .975:
        issues.append("nearly_all_white")
    warnings=[]
    if detail < 4:
        warnings.append("very_low_edge_detail_check_focus")
    if deviation < 12:
        warnings.append("low_tonal_range_review")
    return {
        "pixel_decode_verified":True, "format":actual_format,
        "width":size[0], "height":size[1],
        "sha256":hashlib.sha256(data).hexdigest(),
        "mean_luma":round(mean,2), "luma_std":round(deviation,2),
        "dark_fraction":round(dark_fraction,5),
        "white_fraction":round(white_fraction,5),
        "edge_detail":round(detail,2), "perceptual_dhash":dhash,
        "technical_pass":not issues, "technical_issues":issues,
        "artistic_warnings":warnings,
        "identity_verified":False, "art_review_required":True,
    }


def near_duplicate_checks(scans: list[dict], *, max_distance: int = 2) -> list[dict]:
    """Advisory: dHash near-duplicates across distinct images may be intentional."""
    if type(max_distance) is not int or not 0 <= max_distance <= 64:
        raise ValueError("Invalid dHash distance")
    found=[]
    for i, scan in enumerate(scans):
        if "perceptual_dhash" not in scan:
            continue
        for j in range(i):
            other=scans[j]
            if other.get("perceptual_dhash") is None:
                continue
            distance=(int(scan["perceptual_dhash"],16) ^ int(other["perceptual_dhash"],16)).bit_count()
            if distance <= max_distance:
                found.append({"frame_indices":[j,i],"hamming_distance":distance,
                              "manual_duplicate_review_required":True})
    return found
