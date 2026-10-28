"""Stage Work/illustrator-delivered image layers without rewriting a single story line.

Output is strictly a draft visual composite; only human rights/art review can
mark staged layers approved. Source art is never published automatically.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from studio.identity_qc import load_registry
from studio.premium_assets import inspect_image
from studio.visual_director import planned_shots
from studio.work_contract import script_fingerprint, verify_work_lock

ROLES={"background":-.85,"midground":-.15,"character":.45,
       "foreground":.85,"atmosphere":.95}
SUPPORTED={".png",".jpg",".jpeg",".webp"}

def secure_file(root:Path, path:str)->Path:
    if not isinstance(path,str) or not path or path.startswith(("/","\\")):
        raise ValueError("Source file must be a relative path")
    relative=Path(path)
    if ".." in relative.parts or relative.suffix.lower() not in SUPPORTED:
        raise ValueError("Invalid or unsafe image source")
    absolute=(root/relative).resolve()
    if root.resolve() not in absolute.parents or not absolute.is_file():
        raise ValueError("Source path escapes approved import directory")
    return absolute

def plan_ingestion(ep:dict,ingest:dict,source_root:Path,assets_root:Path,
                   *,registry:dict=None)->tuple[dict,list[tuple[Path,Path]]]:
    verify_work_lock(ep,strict=True)
    if ingest.get("schema")!=1 or ingest.get("episode")!=ep.get("id"):
        raise ValueError("Image map schema/episode mismatch")
    fp=script_fingerprint(ep)
    if ingest.get("script_sha256")!=fp:
        raise ValueError("Image map belongs to a different Work script revision")
    panels=ingest.get("panels")
    if not isinstance(panels,dict):
        raise ValueError("Image map requires panels object")
    sceneids={s["id"] for s in ep["scenes"]}
    if set(panels)!=sceneids:
        raise ValueError("Image map must cover every scene exactly once")
    shots={s["scene"]:s for s in planned_shots(ep,require_work_lock=True)["shots"]}
    registry=registry or load_registry()
    result=copy.deepcopy(ep)
    transfers=[]
    seen_targets=set()
    for scene in result["scenes"]:
        panel=panels[scene["id"]]
        if not isinstance(panel,list) or not 1<=len(panel)<=12:
            raise ValueError("Each panel must contain 1..12 layers")
        if panel[0].get("role")!="background":
            raise ValueError("Panel layers must start with a background")
        layers=[]
        for index,meta in enumerate(panel):
            role=meta.get("role")
            if role not in ROLES:
                raise ValueError("Unknown layer role")
            origin=secure_file(source_root,meta.get("source"))
            original=inspect_image(origin)
            if role!="background" and not original["alpha"]:
                raise ValueError(f"Alpha required for {role}: {origin.name}")
            suffix=origin.suffix.lower()
            target_rel=f"{ep['id']}/{scene['id']}/{index:02}_{role}{suffix}"
            target=(assets_root/target_rel).resolve()
            if assets_root.resolve() not in target.parents or target in seen_targets:
                raise ValueError("Duplicate/unsafe staged target")
            seen_targets.add(target)
            if target.exists():
                raise FileExistsError(f"Will not overwrite existing visual asset: {target}")
            layer={"role":role,"path":target_rel,"depth":ROLES[role],
                   "opacity":1.0,"blend":"over","sha256":original["sha256"],
                   "review_status":"unreviewed","rights":None,
                   "rights_note":"Awaiting documentary license review"}
            if role=="character":
                body=meta.get("body_id")
                if body not in registry["bodies"]:
                    raise ValueError("Character layer must specify a known physical body_id")
                subjects=set(scene.get("subjects",[]))
                center=scene.get("continuity",{}).get("body")
                if center:
                    subjects.add(center)
                if subjects and body not in subjects:
                    raise ValueError("Character illustration not listed in Work script")
                layer["body_id"]=body
                layer["identity_version"]=registry["bodies"][body]["version"]
            layers.append(layer)
            transfers.append((origin,target))
        scene["cinematic"]={"layers":layers,
                           "camera":shots[scene["id"]]["camera"],
                           "vignette":.16,"grain":.03}
    if script_fingerprint(result)!=fp:
        raise RuntimeError("Visual staging unexpectedly changed Work script")
    return result,transfers

def stage(ep_file:Path,map_file:Path,source_root:Path,assets_root:Path,
          output_file:Path)->dict:
    ep=json.loads(ep_file.read_text(encoding="utf-8"))
    ing=json.loads(map_file.read_text(encoding="utf-8"))
    result,transfers=plan_ingestion(ep,ing,source_root,assets_root)
    if output_file.resolve()==ep_file.resolve():
        raise ValueError("Never overwrite the original Work script")
    if output_file.exists():
        raise FileExistsError("Staged file already exists")
    moved=[]
    try:
        for src,dest in transfers:
            dest.parent.mkdir(parents=True,exist_ok=True)
            # Exclusive create prevents replacing previously approved assets.
            with src.open("rb") as source,dest.open("xb") as target:
                shutil.copyfileobj(source,target)
            moved.append(dest)
            if hashlib.sha256(dest.read_bytes()).hexdigest()!=hashlib.sha256(src.read_bytes()).hexdigest():
                raise RuntimeError("Copied artwork changed unexpectedly")
        output_file.parent.mkdir(parents=True,exist_ok=True)
        output_file.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",
                               encoding="utf-8")
    except BaseException:
        for p in moved:
            p.unlink(missing_ok=True)
        raise
    return {"episode":ep["id"],"staged_layers":len(transfers),
            "script_unchanged":True,"art_review_status":"unreviewed",
            "episode_file":str(output_file)}

def main():
    p=argparse.ArgumentParser(description="Stage Work image layers without modifying script")
    p.add_argument("--episode",required=True,type=Path)
    p.add_argument("--image-map",required=True,type=Path)
    p.add_argument("--source-dir",required=True,type=Path)
    p.add_argument("--assets",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    print(json.dumps(stage(args.episode,args.image_map,args.source_dir,
                            args.assets,args.output),indent=2))

if __name__=="__main__":
    main()
