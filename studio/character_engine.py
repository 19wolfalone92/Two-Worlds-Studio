"""Character Animation Engine v1: articulated 2D sprites in one coherent shot.

Input: manually separated RGBA images in a shared, fixed-resolution character
canvas. A hierarchy of pivots and artist-authored keyframes controls root
motion, head turns, arms, feet, gaze layers and fabric. Depth and shadows
belong to the environment, not to a separately floating sticker.

No AI video inference, auto-rigging from a single photo, inverse kinematics,
face landmark detection or photoreal relighting is claimed. This is a real
deterministic keyframed puppet compositor, provided suitable art is supplied.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from studio.premium_assets import resolve_asset, inspect_image

MAX_PARTS=18
MAX_KEYS=80
MAX_PX=12_000_000
ROOT_TRACK="__root__"
KEYS={"at","x","y","angle","scale"}

def _number(v,lo,hi,label):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
        raise ValueError(f"{label} must be finite in {lo}..{hi}")
    return float(v)

def _point(value,label):
    if not isinstance(value,(list,tuple)) or len(value)!=2:
        raise ValueError(f"{label} needs [x,y] normalized coordinates")
    return tuple(_number(v,0,1,label) for v in value)

def _keyframe(frame,label):
    if not isinstance(frame,dict) or "at" not in frame:
        raise ValueError(f"{label} keyframe requires at")
    if set(frame)-KEYS:
        raise ValueError(f"{label}: unsupported keyframe fields")
    return {
        "at":_number(frame["at"],0,1,label+".at"),
        "x":_number(frame.get("x",0),-.6,.6,label+".x"),
        "y":_number(frame.get("y",0),-.6,.6,label+".y"),
        "angle":_number(frame.get("angle",0),-150,150,label+".angle"),
        "scale":_number(frame.get("scale",1),.65,1.5,label+".scale"),
    }

def validate_rig(config:dict,assets_root:Path,canvas_size:tuple[int,int],*,publish=False)->dict:
    if not isinstance(config,dict) or config.get("schema")!=1:
        raise ValueError("Character rig requires schema=1")
    w,h=canvas_size
    if min(w,h)<1 or w*h>MAX_PX:
        raise ValueError("Unsafe character rig canvas size")
    parts=config.get("parts")
    if not isinstance(parts,list) or not 2<=len(parts)<=MAX_PARTS:
        raise ValueError(f"Character rig requires 2..{MAX_PARTS} separate parts")
    names=set()
    parents={}
    sprite_images=[]
    for index,part in enumerate(parts):
        if not isinstance(part,dict):
            raise ValueError("Rig part must be an object")
        name=part.get("name")
        if not isinstance(name,str) or not name.isidentifier() or name.startswith("__") or name in names:
            raise ValueError(f"Duplicate/invalid rig part name {name}")
        names.add(name)
        parent=part.get("parent")
        if parent is not None and (not isinstance(parent,str) or parent==name):
            raise ValueError(f"{name}: invalid parent")
        parents[name]=parent
        _point(part.get("pivot"),f"{name}.pivot")
        _number(part.get("z",index),-100,100,f"{name}.z")
        path=resolve_asset(assets_root,part.get("path"))
        stats=inspect_image(path)
        if (stats["width"],stats["height"])!=(w,h):
            raise ValueError(f"{name}: rig sprite must match full character canvas")
        if not stats["alpha"]:
            raise ValueError(f"{name}: sprite must have real alpha channel")
        if part.get("sha256") is not None and part["sha256"]!=stats["sha256"]:
            raise ValueError(f"{name}: image SHA-256 mismatch")
        if publish:
            if part.get("review_status")!="approved":
                raise ValueError(f"{name}: unapproved rig artwork")
            if part.get("rights") not in ("original","owned","commissioned","licensed"):
                raise ValueError(f"{name}: artwork rights missing")
            if len(str(part.get("rights_note","")).strip())<8:
                raise ValueError(f"{name}: documented asset provenance missing")
            if part.get("sha256")!=stats["sha256"]:
                raise ValueError(f"{name}: master needs SHA-256 pinned sprite")
        sprite_images.append((path,stats))
    for name,parent in parents.items():
        if parent is not None and parent not in names:
            raise ValueError(f"{name}: unknown parent {parent}")
        walked={name}
        while parent is not None:
            if parent in walked:
                raise ValueError(f"{name}: cyclic parent hierarchy")
            walked.add(parent)
            parent=parents[parent]
    tracks=config.get("tracks")
    if not isinstance(tracks,dict) or not tracks:
        raise ValueError("Character rig needs authored motion tracks")
    for name,track in tracks.items():
        if name not in names and name!=ROOT_TRACK:
            raise ValueError(f"Keyframe track targets unknown part {name}")
        if not isinstance(track,list) or not 1<=len(track)<=MAX_KEYS:
            raise ValueError("Each track requires 1..80 frames")
        times=[_keyframe(f,name)["at"] for f in track]
        if times!=sorted(set(times)):
            raise ValueError(f"{name}: unsorted/duplicate track times")
    movement=any(len(v)>1 and any(
        any(abs(_keyframe(f,name)[k]-_keyframe(v[0],name)[k])>.00001
            for k in ("x","y","angle","scale")) for f in v[1:])
        for name,v in tracks.items())
    if publish and not movement:
        raise ValueError("Master character animation must contain changing keyframes")
    if "ground_shadow" in config:
        sh=config["ground_shadow"]
        if not isinstance(sh,dict):
            raise ValueError("ground_shadow must be object")
        _point(sh.get("center"),"ground_shadow.center")
        _point(sh.get("radii"),"ground_shadow.radii")
        _number(sh.get("opacity",.28),0,.65,"ground_shadow.opacity")
        _number(sh.get("blur",.025),0,.12,"ground_shadow.blur")
    if "environment_light" in config:
        light=config["environment_light"]
        if not isinstance(light,dict) or not isinstance(light.get("rgb"),list) or len(light["rgb"])!=3:
            raise ValueError("environment_light needs rgb list")
        for channel in light["rgb"]:
            _number(channel,0,255,"environment_light.rgb")
        _number(light.get("strength",.12),0,.4,"environment_light.strength")
    return {"parts":len(parts),"tracks":len(tracks),"is_acting":movement,
            "needs_human_art_director_review":True}

def _eye():
    return np.eye(3,dtype=np.float64)

def _translate(x,y):
    return np.array([[1,0,x],[0,1,y],[0,0,1]],dtype=np.float64)

def _matrix(k, pivot, canvas):
    angle=math.radians(k["angle"])
    c,s=math.cos(angle),math.sin(angle)
    scale=k["scale"]
    # Image coordinate y increases downward; mathematical positive angle
    # therefore means clockwise on screen.
    rot=np.array([[scale*c,-scale*s,0],[scale*s,scale*c,0],[0,0,1]],dtype=np.float64)
    w,h=canvas
    px,py=pivot
    return _translate(px,py) @ _translate(k["x"]*w,k["y"]*h) @ rot @ _translate(-px,-py)

def _interpolate(frames,at):
    if at<=frames[0]["at"]:
        return frames[0]
    if at>=frames[-1]["at"]:
        return frames[-1]
    for left,right in zip(frames,frames[1:]):
        if left["at"]<=at<=right["at"]:
            d=(at-left["at"])/(right["at"]-left["at"])
            d=d*d*(3-2*d)
            return {key:left[key]*(1-d)+right[key]*d for key in KEYS}
    raise RuntimeError("Cannot interpolate valid rig timeline")

def _shadow(size,config):
    w,h=size
    p=config["ground_shadow"]
    cx,cy=_point(p["center"],"shadow.center")
    rx,ry=_point(p["radii"],"shadow.radii")
    layer=Image.new("RGBA",size,(0,0,0,0))
    draw=ImageDraw.Draw(layer)
    bbox=(round((cx-rx)*w),round((cy-ry)*h),round((cx+rx)*w),round((cy+ry)*h))
    draw.ellipse(bbox,fill=(8,9,15,round(255*p.get("opacity",.28))))
    blur=max(0,round(min(w,h)*p.get("blur",.025)))
    return layer.filter(ImageFilter.GaussianBlur(blur)) if blur else layer

def _transformed_patch(canvas,part,world):
    """Transform only the occupied sprite bounds: vastly faster than per-part full-frame warps."""
    sprite,bounds=part
    left,top,right,bottom=bounds
    cropped=sprite
    h,w=canvas.height,canvas.width
    corners=np.array([[0,right-left,right-left,0],[0,0,bottom-top,bottom-top],
                      [1,1,1,1]],dtype=np.float64)
    m=world@_translate(left,top)
    coords=m@corners
    x0=max(0,int(math.floor(np.min(coords[0])))-2)
    y0=max(0,int(math.floor(np.min(coords[1])))-2)
    x1=min(w,int(math.ceil(np.max(coords[0])))+2)
    y1=min(h,int(math.ceil(np.max(coords[1])))+2)
    if x1<=x0 or y1<=y0:
        return
    inverse=np.linalg.inv(m)
    coeff=(inverse[0,0],inverse[0,1],
           inverse[0,0]*x0+inverse[0,1]*y0+inverse[0,2],
           inverse[1,0],inverse[1,1],
           inverse[1,0]*x0+inverse[1,1]*y0+inverse[1,2])
    patch=cropped.transform((x1-x0,y1-y0),Image.Transform.AFFINE,
                            coeff,resample=Image.Resampling.BICUBIC)
    canvas.alpha_composite(patch,(x0,y0))

class CharacterRig:
    """Load actor sprites once, then render deterministic t=0..1 poses."""
    def __init__(self,config:dict,assets_root:Path,canvas_size:tuple[int,int],
                 *,publish:bool=False):
        self.stats=validate_rig(config,assets_root,canvas_size,publish=publish)
        self.config=config
        self.size=canvas_size
        self.parts={}
        for p in config["parts"]:
            path=resolve_asset(assets_root,p["path"])
            with Image.open(path) as im:
                im.load()
                source=im.convert("RGBA")
            light=config.get("environment_light")
            if light:
                rgba=np.array(source,dtype=np.uint8)
                strength=float(light.get("strength",.12))
                rgb=np.array(light["rgb"],dtype=np.float32)
                rgba[:,:,:3]=np.uint8(np.clip(
                    rgba[:,:,:3].astype(np.float32)*(1-strength)
                    +rgb[None,None,:]*strength,0,255))
                source=Image.fromarray(rgba,"RGBA")
            box=source.getchannel("A").getbbox()
            if box is None:
                raise ValueError(f"{p['name']}: completely transparent sprite")
            self.parts[p["name"]]=(p,(source.crop(box),box))
        self.tracks={name:[_keyframe(x,name) for x in frames]
                     for name,frames in config["tracks"].items()}
        self.shadow=_shadow(canvas_size,config) if config.get("ground_shadow") else None
        self.rest={"at":0.0,"x":0.0,"y":0.0,"angle":0.0,"scale":1.0}

    def pose(self,t:float)->dict:
        t=_number(t,0,1,"animation time")
        memo={}
        root_frame=_interpolate(self.tracks.get(ROOT_TRACK,[self.rest]),t)
        root_pivot=_point(self.config.get("root_pivot",[.5,.78]),"root_pivot")
        root_m=_matrix(root_frame,(root_pivot[0]*self.size[0],
                                   root_pivot[1]*self.size[1]),self.size)
        def world(name):
            if name in memo:
                return memo[name]
            p,_=self.parts[name]
            pivot=_point(p["pivot"],name+".pivot")
            keys=self.tracks.get(name,[self.rest])
            part_m=_matrix(_interpolate(keys,t),
                           (pivot[0]*self.size[0],pivot[1]*self.size[1]),self.size)
            parent=world(p["parent"]) if p.get("parent") else root_m
            memo[name]=parent@part_m
            return memo[name]
        for name in self.parts:
            world(name)
        return memo

    def frame(self,t:float)->Image.Image:
        out=Image.new("RGBA",self.size,(0,0,0,0))
        if self.shadow is not None:
            out.alpha_composite(self.shadow)
        transforms=self.pose(t)
        for name,(part,crop) in sorted(self.parts.items(),
            key=lambda e:(float(e[1][0].get("z",0)),e[0])):
            _transformed_patch(out,crop,transforms[name])
        return out
