"""Read a PRIVATE Work Studio handoff ZIP without copying artwork into a public repo.

This audit is scene/asset integration QA, not a substitute for perceptual
review or a confirmation that finished motion frames are beautiful.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image
from studio.work_contract import script_fingerprint

REQUIRED_NAME="EP01_WORK_CANDIDATE_v1_1.json"
MAX_ENTRY=50_000_000
MAX_ARCHIVE_UNCOMPRESSED=500_000_000

def audit_handoff(path:Path)->dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as z:
        entries=z.infolist()
        if not entries or sum(x.file_size for x in entries)>MAX_ARCHIVE_UNCOMPRESSED:
            raise ValueError("Handoff archive empty or too large")
        files={}
        for i in entries:
            if i.is_dir():
                continue
            p=Path(i.filename)
            if p.is_absolute() or ".." in p.parts or i.file_size>MAX_ENTRY:
                raise ValueError("Unsafe handoff archive path or entry size")
            if p.as_posix() in files:
                raise ValueError("Duplicate file in handoff ZIP")
            files[p.as_posix()]=i
        candidates=[p for p in files if p.endswith("/"+REQUIRED_NAME) or p==REQUIRED_NAME]
        if len(candidates)!=1:
            raise ValueError("Expected exactly one Work candidate episode")
        candidate=candidates[0]
        root=candidate[: -len(REQUIRED_NAME)]
        episode=json.loads(z.read(candidate))
        scenes=episode.get("scenes")
        if not isinstance(scenes,list) or not scenes:
            raise ValueError("Handoff has no scenes")
        expected=episode.get("work_lock",{}).get("script_sha256")
        script_hash=script_fingerprint(episode)
        if expected and expected!=script_hash:
            raise ValueError("Work script fingerprint differs from handoff lock")
        checked=0
        roles={}
        scene_reports=[]
        for scene in scenes:
            config=scene.get("cinematic",{})
            layers=config.get("layers",[])
            if not layers:
                raise ValueError(f"{scene.get('id')}: missing scene art layers")
            character_count=0
            authored_actor_motion=0
            contains_foreground=False
            for layer in layers:
                rel=layer.get("path")
                if not isinstance(rel,str) or rel.startswith("/") or ".." in Path(rel).parts:
                    raise ValueError("Unsafe layer reference")
                path_in_zip=root+"assets/"+rel
                if path_in_zip not in files:
                    raise ValueError(f"Layer missing: {path_in_zip}")
                bytes_=z.read(path_in_zip)
                digest=hashlib.sha256(bytes_).hexdigest()
                if layer.get("sha256") and digest!=layer["sha256"]:
                    raise ValueError(f"Image changed since Work manifest: {rel}")
                try:
                    with Image.open(io.BytesIO(bytes_)) as im:
                        im.load()
                        w,h=im.size
                        has_alpha="A" in im.getbands() or "transparency" in im.info
                except Exception as exc:
                    raise ValueError(f"Cannot decode visual layer: {rel}") from exc
                if w<256 or h<448:
                    raise ValueError(f"Art image too small for even technical preview: {rel}")
                role=layer.get("role")
                roles[role]=roles.get(role,0)+1
                checked+=1
                if role=="character":
                    character_count+=1
                    if not has_alpha:
                        raise ValueError(f"Character lacks transparent cutout: {rel}")
                    if layer.get("motion"):
                        authored_actor_motion+=1
                if role=="foreground":
                    contains_foreground=True
            risks=[]
            if character_count and not authored_actor_motion:
                risks.append("No character animation declared in Studio handoff (Work renderer may have separate motion).")
            if character_count and not scene.get("animation",{}).get("grounding"):
                risks.append("Character contact-shadow/occlusion/lighting integration not documented.")
            if character_count and not contains_foreground:
                risks.append("No foreground occluder: check whether actor looks pasted onto background.")
            for layer in layers:
                if layer.get("motion",{}).get("blink") and not layer["motion"]["blink"].get("erase_mask"):
                    risks.append("Unsafe blink overlay missing eye-erasure mask; Studio will freeze blinking.")
            scene_reports.append({
                "id":scene.get("id"),"layer_count":len(layers),
                "characters":character_count,"authored_actor_motion":authored_actor_motion,
                "risks":risks,
            })
    return {
        "source":path.name,"episode":episode.get("id"),
        "scenes":len(scenes),"visual_assets_checked":checked,
        "roles":roles,"work_script_hash_verified":bool(expected),
        "studio_character_motion_coverage":sum(s["authored_actor_motion"] for s in scene_reports),
        "scenes_with_sticker_risk":sum(any("pasted" in r or "integration" in r for r in s["risks"]) for s in scene_reports),
        "scene_reports":scene_reports,
        "professional_animation_approved":False,
        "safe_for_public_repo_upload":False,
        "note":"Structural preflight only. Artwork, physical integration and final film require human review."
    }

def main():
    p=argparse.ArgumentParser(description="Audit real Work handoff ZIP without publishing its contents")
    p.add_argument("--handoff",required=True,type=Path)
    p.add_argument("--report",type=Path)
    args=p.parse_args()
    result=json.dumps(audit_handoff(args.handoff),indent=2,ensure_ascii=False)+"\n"
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(result,encoding="utf-8")
    print(result)

if __name__=="__main__":
    main()
