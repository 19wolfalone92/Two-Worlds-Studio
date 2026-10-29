"""Identity continuity for FICTIONAL character bodies, using approved artwork references.

A registry of reference images is a production contract, not a face-recognition
claim. The engine verifies matching body ID, version and asset hashes across
shots. It cannot certify facial similarity or anatomy without visual review.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from studio.premium_assets import resolve_asset, inspect_image

ROOT=Path(__file__).resolve().parent.parent
REGISTRY=ROOT/"story/identity_registry.json"
VALID_VIEWS={"front","three_quarter","profile","full_body","expression",
             "back","rear_three_quarter","hands","blink"}
VALID_RIGHTS={"original","owned","commissioned","licensed"}

def load_registry(path:Path=REGISTRY)->dict:
    data=json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema")!=1 or not isinstance(data.get("bodies"),dict):
        raise ValueError("Character registry needs schema=1 and bodies object")
    for body,details in data["bodies"].items():
        if not body or not isinstance(details,dict) or not details.get("version"):
            raise ValueError(f"{body}: invalid identity registry entry")
        references=details.get("references",[])
        if not isinstance(references,list):
            raise ValueError(f"{body}: references must be a list")
        for ref in references:
            if (not isinstance(ref,dict) or ref.get("view") not in VALID_VIEWS
                    or ref.get("review_status") not in ("approved","unreviewed")
                    or not isinstance(ref.get("file"),str)):
                raise ValueError(f"{body}: bad reference image metadata")
    return data

def verify_reference(ref:dict,assets:Path,*,strict:bool)->dict:
    p=resolve_asset(assets,ref["file"])
    stats=inspect_image(p)
    if stats["sha256"]!=ref.get("sha256"):
        raise ValueError(f"Reference changed or missing SHA-256: {ref['file']}")
    if strict:
        if ref.get("review_status")!="approved":
            raise ValueError(f"{ref['file']}: character reference not artist-approved")
        if ref.get("rights") not in VALID_RIGHTS or len(ref.get("rights_note","").strip())<8:
            raise ValueError(f"{ref['file']}: insufficient reference rights provenance")
        if min(stats["width"],stats["height"])<512:
            raise ValueError(f"{ref['file']}: reference too small")
    return {"file":ref["file"],"view":ref["view"],
            "sha256":stats["sha256"],"approved":ref.get("review_status")=="approved"}

def verify_body(body_id:str,version:str,registry:dict,assets:Path,*,strict=True)->dict:
    body=registry["bodies"].get(body_id)
    if body is None:
        raise ValueError(f"Unknown physical body {body_id}")
    if version!=body["version"]:
        raise ValueError(f"{body_id}: version drift: {version} != {body['version']}")
    references=body.get("references",[])
    if strict and len(references)<3:
        raise ValueError(f"{body_id}: approved front/three-quarter/profile reference set missing")
    checked=[verify_reference(x,assets,strict=strict) for x in references]
    if strict:
        if len({x["sha256"] for x in checked})<len(checked):
            raise ValueError(f"{body_id}: duplicate image used for multiple visual reference angles")
        needed={"front","three_quarter","profile"}
        views={x["view"] for x in checked if x["approved"]}
        if not needed<=views:
            raise ValueError(f"{body_id}: missing approved reference angles {sorted(needed-views)}")
    return {"body":body_id,"version":version,"checked":len(checked),
            "visual_match_verified":False}

def verify_scene_characters(scene:dict,registry:dict,assets:Path,*,strict=True)->dict:
    config=scene.get("cinematic")
    if not isinstance(config,dict):
        if strict:
            raise ValueError(f"{scene.get('id')}: no premium layers")
        return {"scene":scene.get("id"),"characters":[]}
    subjects=set(scene.get("subjects",[]))
    annotated=scene.get("continuity",{}).get("body")
    if annotated:
        subjects.add(annotated)
    character_layers=[x for x in config.get("layers",[]) if x.get("role")=="character"]
    if strict and subjects and not character_layers:
        raise ValueError(f"{scene.get('id')}: character artwork missing for script subjects")
    findings=[]
    depicted=set()
    for layer in character_layers:
        body_id=layer.get("body_id")
        if not isinstance(body_id,str) or not body_id:
            raise ValueError(f"{scene.get('id')}: every character layer requires body_id")
        if body_id in depicted:
            raise ValueError(f"{scene.get('id')}: duplicate body assigned to multiple layers")
        depicted.add(body_id)
        if strict and subjects and body_id not in subjects:
            raise ValueError(f"{scene.get('id')}: visual body {body_id} not in script subjects")
        findings.append(verify_body(body_id,layer.get("identity_version"),
                                    registry,assets,strict=strict))
    if strict and subjects and not subjects<=depicted:
        raise ValueError(f"{scene.get('id')}: missing character layer for {sorted(subjects-depicted)}")
    return {"scene":scene.get("id"),"characters":findings,
            "identity_visual_inspection_required":True}

def verify_episode_characters(episode:dict,assets:Path,*,registry_path:Path=REGISTRY,
                              strict=True)->dict:
    registry=load_registry(registry_path)
    results=[verify_scene_characters(scene,registry,assets,strict=strict)
             for scene in episode.get("scenes",[])]
    return {"episode":episode.get("id"),"scenes":results,
            "human_identity_review_required":True}

def main():
    p=argparse.ArgumentParser(description="Physical-body reference asset gate")
    p.add_argument("--episode",required=True,type=Path)
    p.add_argument("--registry",type=Path,default=REGISTRY)
    p.add_argument("--assets",type=Path,default=ROOT/"assets")
    p.add_argument("--strict",action="store_true")
    args=p.parse_args()
    episode=json.loads(args.episode.read_text(encoding="utf-8"))
    print(json.dumps(verify_episode_characters(episode,args.assets,
             registry_path=args.registry,strict=args.strict),indent=2))

if __name__=="__main__":
    main()
