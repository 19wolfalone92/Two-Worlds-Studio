"""Fail-closed, no-spend-by-default Cloudflare FLUX.2 [klein] image handoff.

A confirmed API token does NOT prove that image inference, daily remaining quota,
visual quality, or character consistency works. An invocation generates at most
one unreviewed PNG and never uploads it to a public GitHub artifact.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MODEL = "@cf/black-forest-labs/flux-2-klein-4b"
BASE = "https://api.cloudflare.com/client/v4/accounts/"
ROOT = Path(__file__).resolve().parent.parent
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,64}$")
ACCOUNT_ID = re.compile(r"^[A-Fa-f0-9]{32}$")
MAX_REFERENCE_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 30 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024


class FreeImageError(RuntimeError):
    """A sanitized provider failure, never a raw API response or request URL."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_dimensions(width: int, height: int) -> None:
    if (type(width) is not int or type(height) is not int
            or not 256 <= width <= 1920 or not 256 <= height <= 1920):
        raise ValueError("Cloudflare image dimensions must be integers from 256 to 1920")


def checked_reference(path: Path) -> tuple[bytes, str]:
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Reference must be a PNG/JPEG/WebP file")
    if not path.is_file() or not 0 < path.stat().st_size < MAX_REFERENCE_BYTES:
        raise ValueError("Missing or oversized private reference file")
    payload = path.read_bytes()
    with Image.open(io.BytesIO(payload)) as im:
        im.verify()
    with Image.open(io.BytesIO(payload)) as im:
        if im.width >= 512 or im.height >= 512:
            raise ValueError("Cloudflare FLUX.2 reference dimensions must be smaller than 512x512")
        if im.format not in ("PNG", "JPEG", "WEBP"):
            raise ValueError("Unsupported reference image encoding")
        kind = im.format
    return payload, {"PNG": "image/png", "JPEG": "image/jpeg",
                     "WEBP": "image/webp"}[kind]


def multipart(prompt: str, width: int, height: int, seed: int,
              references: list[Path]) -> tuple[bytes, str, list[str]]:
    """Construct a fixed-host multipart payload; no filenames or URLs from user input."""
    if not isinstance(prompt, str) or not 10 <= len(prompt.strip()) <= 8000:
        raise ValueError("Expected a nonempty bounded visual prompt")
    validate_dimensions(width, height)
    if type(seed) is not int or not 0 <= seed < 2147483648:
        raise ValueError("Seed must be a nonnegative 31-bit integer")
    if len(references) > 4:
        raise ValueError("FLUX.2 supports no more than four reference images")
    boundary = ("----TwoWorlds" + uuid.uuid4().hex).encode("ascii")
    out = bytearray()
    sha = []

    def field(name: str, value: bytes, mimetype: str | None = None) -> None:
        out.extend(b"--" + boundary + b"\r\n")
        out.extend(('Content-Disposition: form-data; name="' + name + '"').encode("ascii"))
        if mimetype:
            ext = {"image/png": ".png", "image/jpeg": ".jpg",
                   "image/webp": ".webp"}[mimetype]
            out.extend(('; filename="' + name + ext + '"').encode("ascii"))
        out.extend(b"\r\n")
        if mimetype:
            out.extend(('Content-Type: ' + mimetype).encode("ascii") + b"\r\n")
        out.extend(b"\r\n")
        out.extend(value)
        out.extend(b"\r\n")

    field("prompt", prompt.encode("utf-8"))
    field("width", str(width).encode("ascii"))
    field("height", str(height).encode("ascii"))
    field("seed", str(seed).encode("ascii"))
    for i, reference in enumerate(references):
        payload, mimetype = checked_reference(reference)
        field(f"input_image_{i}", payload, mimetype)
        sha.append(digest(payload))
    out.extend(b"--" + boundary + b"--\r\n")
    return bytes(out), "multipart/form-data; boundary=" + boundary.decode("ascii"), sha


def parse_image_response(payload: bytes, width: int, height: int) -> bytes:
    """Decode Cloudflare's JSON base64 result, verify pixels, normalize to PNG."""
    if len(payload) > MAX_RESPONSE_BYTES:
        raise FreeImageError("Provider response exceeded image-size limit")
    try:
        response = json.loads(payload)
    except (ValueError, UnicodeError):
        raise FreeImageError("Provider response was not valid JSON") from None
    if not isinstance(response, dict) or response.get("success") is False:
        raise FreeImageError("Provider returned unsuccessful image generation")
    result = response.get("result", response)
    if not isinstance(result, dict) or not isinstance(result.get("image"), str):
        raise FreeImageError("Provider returned no base64 image")
    encoded = result["image"]
    if encoded.startswith("data:image/"):
        encoded = encoded.split(",", 1)[-1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise FreeImageError("Provider image is not valid base64") from None
    if not 100 < len(raw) < MAX_IMAGE_BYTES:
        raise FreeImageError("Provider image is empty or oversized")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            if im.width != width or im.height != height:
                raise FreeImageError("Provider image dimensions do not match requested dimensions")
            if im.format not in ("PNG", "JPEG", "WEBP"):
                raise FreeImageError("Provider returned unsupported image format")
            normalized = im.convert("RGB")
            buf = io.BytesIO()
            normalized.save(buf, format="PNG", optimize=False)
    except (OSError, UnidentifiedImageError):
        raise FreeImageError("Provider returned an undecodable image") from None
    return buf.getvalue()


def request_image(*, token: str, account_id: str, prompt: str,
                  width: int, height: int, seed: int, references: list[Path],
                  opener=urllib.request.urlopen) -> tuple[bytes, list[str]]:
    if not token:
        raise ValueError("CLOUDFLARE_AI_TOKEN is not configured")
    if not ACCOUNT_ID.fullmatch(account_id or ""):
        raise ValueError("CLOUDFLARE_ACCOUNT_ID is missing or invalid")
    data, content_type, reference_shas = multipart(prompt, width, height, seed, references)
    request = urllib.request.Request(
        BASE + account_id + "/ai/run/" + MODEL,
        data=data, method="POST",
        headers={"Authorization": "Bearer " + token, "Content-Type": content_type,
                 "Accept": "application/json"})
    try:
        with opener(request, timeout=100) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise FreeImageError("Cloudflare image API rate limited the request (HTTP 429)") from None
        raise FreeImageError(f"Cloudflare image API returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise FreeImageError("Cloudflare image API network or timeout failure") from None
    return parse_image_response(payload, width, height), reference_shas


def plan(shot: dict, width: int = 1024, height: int = 1792,
         references: list[Path] | None = None) -> dict:
    """Default path: no token read and no network activity."""
    if not isinstance(shot, dict) or not SAFE_ID.fullmatch(str(shot.get("scene", ""))):
        raise ValueError("A safe scene ID is required")
    prompt = shot.get("prompt")
    if not isinstance(prompt, str) or not 10 <= len(prompt.strip()) <= 8000:
        raise ValueError("Scene must contain a bounded visual prompt")
    validate_dimensions(width, height)
    seed = shot.get("seed")
    if type(seed) is not int or not 0 <= seed < 2147483648:
        raise ValueError("Scene requires an explicit deterministic seed")
    if references is None:
        references = []
    # Reject oversized or unsupported references before any remote request.
    reference_shas = [digest(checked_reference(ref)[0]) for ref in references]
    if len(references) > 4:
        raise ValueError("At most four references")
    return {"schema": 1, "provider": "cloudflare_workers_ai", "model": MODEL,
            "scene": shot["scene"], "seed": seed, "width": width, "height": height,
            "reference_sha256": reference_shas,
            "prompt_sha256": digest(prompt.encode("utf-8")),
            "review_status": "unreviewed", "ready_for_publication": False,
            "image_generated": False, "estimate_only": True}


def render_one(shot: dict, destination: Path, *, token: str, account_id: str,
               width: int = 1024, height: int = 1792,
               references: list[Path] | None = None, client=request_image) -> dict:
    """Single explicit opt-in image call; rejects any existing target before spending quota."""
    references = list(references or [])
    proposal = plan(shot, width, height, references)
    output_dir = destination.resolve()
    if output_dir == ROOT or ROOT in output_dir.parents:
        raise ValueError("Never stage unpublished art inside the public source repository")
    name = shot["scene"]
    if not SAFE_ID.fullmatch(name):
        raise ValueError("Invalid scene ID")
    image = output_dir / (name + ".png")
    receipt = output_dir / (name + ".art_review.json")
    if image.exists() or receipt.exists():
        raise FileExistsError("Unreviewed art exists; do not spend quota regenerating it")
    # Single image per invocation. No implicit paid provider or bulk fallback.
    payload, shas = client(token=token, account_id=account_id, prompt=shot["prompt"],
                           width=width, height=height, seed=shot["seed"],
                           references=references)
    # Client mocks may return raw PNG, but it still must be a real image.
    with Image.open(io.BytesIO(payload)) as test:
        test.verify()
    with Image.open(io.BytesIO(payload)) as test:
        if test.format != "PNG" or test.size != (width, height):
            raise FreeImageError("Final output is not a real PNG at requested resolution")
    if shas != proposal["reference_sha256"]:
        raise FreeImageError("Provider reference fingerprints changed unexpectedly")
    # Inspect actual pixels before staging, not only the PNG header.
    from studio.art_image_qc import inspect_image_bytes
    image_qc = inspect_image_bytes(payload, expected_size=(width, height))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(proposal)
    metadata.update(image_generated=True, estimate_only=False,
                    png_sha256=digest(payload), rights_review_required=True,
                    visual_identity_review_required=True,
                    technical_image_qc=image_qc, published=False,
                    master_ready=False)
    # Reserve exclusive files: existing files must not be replaced.
    try:
        with image.open("xb") as f:
            f.write(payload)
        with receipt.open("x", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except BaseException:
        image.unlink(missing_ok=True)
        receipt.unlink(missing_ok=True)
        raise
    return metadata


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Prepare or opt-in to ONE Cloudflare FLUX art frame")
    p.add_argument("--prompt-pack", type=Path, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/two_worlds_private_art"))
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1792)
    p.add_argument("--reference", type=Path, action="append", default=[])
    p.add_argument("--execute-one", action="store_true",
                   help="Explicitly spend one Cloudflare Workers AI inference; never bulk")
    args = p.parse_args(argv)
    pack = json.loads(args.prompt_pack.read_text(encoding="utf-8"))
    scenes = pack.get("scenes")
    if not isinstance(scenes, list):
        p.error("Prompt pack must have a scenes array")
    selected = [scene for scene in scenes if isinstance(scene, dict)
                and scene.get("scene") == args.scene]
    if len(selected) != 1:
        p.error("Selected scene must appear exactly once")
    shot = selected[0]
    if not args.execute_one:
        report = plan(shot, args.width, args.height, args.reference)
    else:
        report = render_one(shot, args.output_dir,
             token=os.environ.get("CLOUDFLARE_AI_TOKEN", ""),
             account_id=os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""),
             width=args.width, height=args.height, references=args.reference)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
