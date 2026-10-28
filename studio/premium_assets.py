"""Production artwork validation: file integrity, layers, rights and visual preflight.

No model inference. A machine cannot prove artistic quality or copyright ownership;
this module makes provenance and technical defects visible before approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

ROLES=("background","midground","character","foreground","atmosphere")
COMPOSITES=("over","screen","add")
LICENSES=("original","owned","commissioned","licensed")
MAX_IMAGE_BYTES=45*1024*1024
MAX_TOTAL_PIXELS=80_000_000


def resolve_asset(root:Path, relative:str)->Path:
    """A relative path strictly inside assets, never a symlink escaping the root."""
    if not isinstance(relative,str) or not relative or relative.startswith(("/", "\\")):
        raise ValueError("Asset must be a non-empty relative path")
    p=Path(relative)
    if ".." in p.parts or p.suffix.lower() not in (".png",".jpg",".jpeg",".webp"):
        raise ValueError(f"Invalid artwork path: {relative}")
    resolved=(root/relative).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError("Artwork path escapes asset directory")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if resolved.stat().st_size>MAX_IMAGE_BYTES:
        raise ValueError("Artwork exceeds file-size limit")
    return resolved


def inspect_image(path:Path)->dict:
    try:
        with Image.open(path) as im:
            if im.width*im.height>MAX_TOTAL_PIXELS:
                raise ValueError("Artwork exceeds decoded pixel limit")
            im.load()
            w,h=im.size
            mode=im.mode
            has_alpha=("A" in im.getbands()) or "transparency" in im.info
            sample=im.convert("RGBA")
            sample.thumbnail((256,448))
            pixels=np.asarray(sample,dtype=np.float32)/255.0
            alpha=pixels[:,:,3]
            visible=alpha>.05
            if np.any(visible):
                colors=pixels[:,:,:3][visible]
                brightness=float(colors.mean())
                contrast=float(colors.mean(axis=1).std())
            else:
                brightness=0.0
                contrast=0.0
            gray=np.asarray(sample.convert("L"),dtype=np.float32)
            detail=float(np.mean(np.abs(np.diff(gray,axis=1))))
            alpha_coverage=float(alpha.mean()) if has_alpha else 1.0
            return {"width":w,"height":h,"mode":mode,"alpha":has_alpha,
                    "brightness":round(brightness,4),"contrast":round(contrast,4),
                    "edge_activity":round(detail,3),"alpha_coverage":round(alpha_coverage,4),
                    "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    except (OSError,UnidentifiedImageError) as exc:
        raise ValueError(f"Unreadable image: {path}") from exc


def validate_layer(layer:dict, root:Path, *, publish:bool=False, min_width:int=256,
                   min_height:int=448)->tuple[dict,list[str]]:
    if not isinstance(layer,dict) or layer.get("role") not in ROLES:
        raise ValueError("Every layer needs a supported role")
    if layer.get("blend","over") not in COMPOSITES:
        raise ValueError("Unsupported blend mode")
    depth=layer.get("depth",0.0)
    if isinstance(depth,bool) or not isinstance(depth,(float,int)) or not -1<=depth<=1:
        raise ValueError("Parallax depth must be between -1 and +1")
    opacity=layer.get("opacity",1.0)
    if isinstance(opacity,bool) or not isinstance(opacity,(float,int)) or not 0<=opacity<=1:
        raise ValueError("Layer opacity must be between 0 and 1")
    if "motion" in layer:
        if layer["role"]!="character" or "rig" in layer:
            raise ValueError("Legacy actor motion cannot be combined with an articulated rig")
        from studio.actor_motion import validate_motion
        validate_motion(layer["motion"],root,publish=publish)
    path=resolve_asset(root,layer.get("path"))
    result=inspect_image(path)
    if "rig" in layer:
        if layer["role"]!="character":
            raise ValueError("Only character layers support articulated rig")
        from studio.character_engine import validate_rig
        validate_rig(layer["rig"],root,(result["width"],result["height"]),publish=publish)
    warnings=[]
    if result["width"]<min_width or result["height"]<min_height:
        warnings.append("Artwork resolution below target for chosen output")
    if layer["role"]!="background" and not result["alpha"]:
        warnings.append("Non-background layer has no alpha channel; may obscure lower layers")
    if layer["role"]=="background" and result["contrast"]<0.025:
        warnings.append("Very low background luminance contrast — check visibility on mobile")
    if layer["role"]=="background" and (result["brightness"]<0.065 or result["brightness"]>0.96):
        warnings.append("Extreme background brightness — check exposure")
    if result["alpha"] and result["alpha_coverage"]<0.002:
        warnings.append("Layer is almost completely transparent")
    if layer.get("sha256") is not None and layer["sha256"]!=result["sha256"]:
        raise ValueError(f"Artwork hash mismatch: {layer['path']}")
    if publish:
        if layer.get("review_status")!="approved":
            raise ValueError(f"{layer['path']}: image is not explicitly approved")
        if layer.get("rights") not in LICENSES:
            raise ValueError(f"{layer['path']}: missing acceptable rights declaration")
        if not isinstance(layer.get("rights_note"),str) or len(layer["rights_note"].strip())<8:
            raise ValueError(f"{layer['path']}: rights provenance note missing")
        if result["width"]<min_width or result["height"]<min_height:
            raise ValueError(f"{layer['path']}: production image resolution too low")
        if layer["role"]!="background" and not result["alpha"]:
            raise ValueError(f"{layer['path']}: expected transparent PNG/WebP foreground")
    return {"path":layer["path"],"role":layer["role"],"depth":float(depth),
            "opacity":float(opacity),"blend":layer.get("blend","over"),
            **result},warnings


def validate_premium_scene(scene:dict, root:Path, *, publish:bool=False,
                           min_width:int=256,min_height:int=448)->dict:
    config=scene.get("cinematic")
    if config is None:
        if publish:
            raise ValueError(f"{scene.get('id')}: premium release requires cinematic scene layers")
        return {"scene":scene.get("id"),"cinematic":False,"layers":[],
                "warnings":["Procedural placeholder art, not production approved"],
                "tech_preflight_pass":False}
    if not isinstance(config,dict) or not isinstance(config.get("layers"),list):
        raise ValueError("cinematic needs layers list")
    layers=config["layers"]
    if not 1<=len(layers)<=12:
        raise ValueError("cinematic needs 1..12 layers")
    if layers[0].get("role")!="background":
        raise ValueError("First cinematic layer must be background")
    assets=[]
    warnings=[]
    for item in layers:
        asset,notes=validate_layer(item,root,publish=publish,
                                   min_width=min_width,min_height=min_height)
        assets.append(asset)
        warnings += [f"{item['path']}: {note}" for note in notes]
    camera=config.get("camera",{})
    if not isinstance(camera,dict):
        raise ValueError("cinematic.camera must be object")
    for key in ("start","end"):
        state=camera.get(key,{"x":0.0,"y":0.0,"zoom":1.0})
        if not isinstance(state,dict):
            raise ValueError("Camera keyframe must be an object")
        for coordinate in ("x","y"):
            v=state.get(coordinate,0.0)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or abs(v)>1:
                raise ValueError("Camera x,y values must be normalized -1..+1")
        z=state.get("zoom",1.0)
        if isinstance(z,bool) or not isinstance(z,(int,float)) or not 1<=z<=1.4:
            raise ValueError("Camera zoom must be within 1..1.4")
    if config.get("grain",0.0) not in (0,0.0,None):
        g=config["grain"]
        if isinstance(g,bool) or not isinstance(g,(int,float)) or not 0<=g<=0.25:
            raise ValueError("Film grain 0..0.25")
    return {"scene":scene.get("id"),"cinematic":True,
            "layers":assets,"warnings":warnings,
            "tech_preflight_pass":not warnings}


def review_episode(episode:dict, root:Path, *, publish:bool=False,
                   min_width:int=1080,min_height:int=1920)->dict:
    scenes=episode.get("scenes")
    if not isinstance(scenes,list) or not scenes:
        raise ValueError("Episode has no scenes")
    results=[validate_premium_scene(s,root,publish=publish,
                                   min_width=min_width,min_height=min_height) for s in scenes]
    if publish and not all(v["tech_preflight_pass"] for v in results):
        raise ValueError("Premium technical quality warnings must be reviewed and fixed before approval")
    return {"episode":episode.get("id"),"publish_gate":publish,
            "cinematic_panels":sum(1 for v in results if v["cinematic"]),
            "total_scenes":len(results),"results":results,
            "technical_pass":all(r["tech_preflight_pass"] for r in results)}


def main():
    p=argparse.ArgumentParser(description="Review cinematic art inputs; never automatically approves art")
    p.add_argument("--episode",type=Path,required=True)
    p.add_argument("--assets",type=Path,default=Path(__file__).resolve().parent.parent/"assets")
    p.add_argument("--report",type=Path,default=None)
    p.add_argument("--publish-gate",action="store_true",
                   help="Require approved, rights-cleared, 1080x1920 art in EVERY scene")
    args=p.parse_args()
    ep=json.loads(args.episode.read_text(encoding="utf-8"))
    report=review_episode(ep,args.assets,publish=args.publish_gate)
    output=json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(output,encoding="utf-8")
    print(output)


if __name__=="__main__":
    main()
