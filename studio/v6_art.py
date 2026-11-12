"""Private visual contact sheet and reference-graph preflight for Studio V6.

This creates a review IMAGE, not a rendered movie or AI generation.
It never changes approved artwork, edits prompts or spends GPU quota.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from studio.v6_production import (
    CAST, SCENES, ProductionBlocked, asset, private_root, read_storyboard,
)


def reference_graph(path: Path) -> dict:
    if path.stat().st_size > 2 * 1024**2:
        raise ProductionBlocked("Oversized ComfyUI API graph")
    graph = json.loads(path.read_text(encoding="utf-8"))
    from studio.comfy_reference_guard import audit_reference_conditioning
    result = audit_reference_conditioning(graph)
    if result.get("topological_reference_path") is not True:
        raise ProductionBlocked("Reference image does not actually condition the output")
    return {"status": "REFERENCE_GRAPH_STRUCTURALLY_CONDITIONED",
            **result, "actual_character_identity_verified": False,
            "gpu_executed": False, "published": False}


def contact_sheet(storyboard: Path, root_path: Path, output_name: str) -> dict:
    story = read_storyboard(storyboard)
    root = private_root(root_path)
    if not output_name.endswith(".jpg") or "/" in output_name or "\\" in output_name or ".." in output_name:
        raise ProductionBlocked("Contact sheet output must be a new private .jpg filename")
    dest = root / output_name
    if dest.exists() or dest.is_symlink():
        raise ProductionBlocked("Refuse to overwrite previously reviewed storyboards")
    items = []
    for who in CAST:
        path = asset(root, story["references"][who]["image"], (".png",))
        items.append(("CANON: " + who.upper(), path))
    for shot in story["shots"]:
        path = asset(root, shot["image"], (".png",))
        items.append((shot["id"] + ": " + ",".join(shot["characters"]).upper(), path))
    tile_w, tile_h, bar = 246, 440, 42
    canvas = Image.new("RGB", (tile_w * 3, (tile_h + bar) * 3), "#13141a")
    drawer = ImageDraw.Draw(canvas)
    for i, (label, path) in enumerate(items):
        with Image.open(path) as image:
            if image.width < 576 or image.height < 1024:
                raise ProductionBlocked("Art too small for publication keyframe: " + label)
            thumb = ImageOps.fit(image.convert("RGB"), (tile_w, tile_h),
                                 method=Image.Resampling.LANCZOS)
        left = (i % 3) * tile_w
        top = (i // 3) * (tile_h + bar)
        canvas.paste(thumb, (left, top))
        drawer.rectangle((left, top + tile_h, left + tile_w, top + tile_h + bar), fill="#25222a")
        drawer.text((left + 9, top + tile_h + 9), label, fill="#f8d9c1")
    canvas.save(dest, format="JPEG", quality=88, optimize=True)
    return {"status": "NINE_IMAGE_CONTACT_SHEET_FOR_HUMAN_REVIEW",
            "output": str(dest), "characters": 2, "scenes": len(SCENES),
            "visual_approval": False, "gpu_executed": False, "published": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode", required=True)
    x=sub.add_parser("reference-graph")
    x.add_argument("--api-graph", required=True, type=Path)
    y=sub.add_parser("contact-sheet")
    y.add_argument("--storyboard", required=True, type=Path)
    y.add_argument("--assets", required=True, type=Path)
    y.add_argument("--output", default="EB01_VISUAL_REVIEW_NOT_APPROVED.jpg")
    a=p.parse_args(argv)
    if a.mode=="reference-graph":
        result=reference_graph(a.api_graph)
    else:
        result=contact_sheet(a.storyboard, a.assets, a.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
