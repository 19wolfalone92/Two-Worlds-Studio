"""Deterministic subtle 2D actor animation driven by approved sprite layers.

For real facial animation artists must supply blink/expression overlays drawn for
the same physical body and coordinate space. Do not synthesize new faces here.
"""
from __future__ import annotations
import math
from pathlib import Path
from PIL import Image, ImageChops
from studio.premium_assets import resolve_asset,inspect_image

def validate_motion(meta:dict,assets:Path,*,publish:bool=False)->dict:
    if not isinstance(meta,dict):
        raise ValueError("Character motion must be object")
    allowed={"breath_scale":(.0,.03),"sway_x":(.0,.05),"sway_y":(.0,.05),
             "cycles":(.2,6)}
    for key,(lo,hi) in allowed.items():
        n=meta.get(key,1.0 if key=="cycles" else 0.0)
        if isinstance(n,bool) or not isinstance(n,(int,float)) or not lo<=n<=hi:
            raise ValueError(f"{key} must be {lo}..{hi}")
    blink=meta.get("blink")
    result={"blink_frames":0}
    if blink is not None:
        if not isinstance(blink,dict):
            raise ValueError("Blink overlay must be object")
        frames=blink.get("at",[])
        if not isinstance(frames,list) or any(isinstance(t,bool) or
             not isinstance(t,(int,float)) or not 0<t<1 for t in frames):
            raise ValueError("Blink times must be normalized values between 0 and 1")
        duration=blink.get("duration",.045)
        if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not 0.015<=duration<=.15:
            raise ValueError("Blink duration must be .015..15 normalized")
        overlay=resolve_asset(assets,blink.get("path"))
        info=inspect_image(overlay)
        if not info["alpha"]:
            raise ValueError("Blink overlay needs transparent alpha image")
        if blink.get("sha256") and blink["sha256"]!=info["sha256"]:
            raise ValueError("Blink overlay sha mismatch")
        if publish:
            if blink.get("review_status")!="approved" or blink.get("rights") not in (
                "original","owned","commissioned","licensed") or len(blink.get("rights_note","").strip())<8:
                raise ValueError("Blink overlay has not been rights/artist approved")
        result={"blink_frames":len(frames),"overlay":blink["path"]}
    return result

def animate_actor(source:Image.Image,meta:dict,t:float,*,overlay:Image.Image=None)->Image.Image:
    src=source.convert("RGBA")
    w,h=src.size
    t=max(0.0,min(1.0,float(t)))
    cycles=float(meta.get("cycles",1.0))
    phase=2*math.pi*t*cycles
    scale=1+float(meta.get("breath_scale",0.0))*math.sin(phase)
    dx=round(w*float(meta.get("sway_x",0))*math.sin(phase))
    dy=round(h*float(meta.get("sway_y",0))*math.cos(phase))
    zw,zh=round(w*scale),round(h*scale)
    if zw!=w or zh!=h:
        src=src.resize((max(1,zw),max(1,zh)),Image.Resampling.BICUBIC)
    # Transparent canvas avoids wraparound artifacts from ImageChops.offset.
    target=Image.new("RGBA",(w,h),(0,0,0,0))
    target.alpha_composite(src,(round((w-zw)/2)+dx,round((h-zh)/2)+dy))
    if overlay is not None and meta.get("blink"):
        schedule=meta["blink"].get("at",[])
        duration=float(meta["blink"].get("duration",.045))
        if any(abs(t-float(at))<duration/2 for at in schedule):
            closed=overlay.convert("RGBA").resize((max(1,zw),max(1,zh)),Image.Resampling.BICUBIC)
            layer=Image.new("RGBA",(w,h),(0,0,0,0))
            layer.alpha_composite(closed,(round((w-zw)/2)+dx,round((h-zh)/2)+dy))
            target=Image.alpha_composite(target,layer)
    return target
