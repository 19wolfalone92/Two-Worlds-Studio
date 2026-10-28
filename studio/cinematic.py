"""Layered 2.5D motion-comic compositor.

Takes approved multi-depth PNG/WebP/JPEG layers, NOT AI prompts. Animates a
repeatable eased camera, parallax, atmospheric lights, and gentle film texture.
No GPU or external APIs required. The existing renderer adds effects/titles
and encodes H.264/AAC MP4.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

try:
    from studio.premium_assets import resolve_asset, validate_premium_scene
    from studio.actor_motion import animate_actor, validate_motion
except ModuleNotFoundError:
    from premium_assets import resolve_asset, validate_premium_scene
    from actor_motion import animate_actor, validate_motion


def ease(x:float)->float:
    x=max(0.0,min(1.0,float(x)))
    return x*x*(3-2*x)


def camera_state(scene:dict,progress:float)->tuple[float,float,float]:
    camera=scene["cinematic"].get("camera",{})
    start=camera.get("start",{"x":0,"y":0,"zoom":1})
    end=camera.get("end",{"x":0,"y":0,"zoom":1.08})
    t=ease(progress)
    value=lambda key,default: float(start.get(key,default))*(1-t)+float(end.get(key,default))*t
    return value("x",0),value("y",0),value("zoom",1.0)


def _cover_rgba(im:Image.Image,size:tuple[int,int])->Image.Image:
    return ImageOps.fit(im.convert("RGBA"),size,method=Image.Resampling.LANCZOS,
                        centering=(0.5,0.5))


def _place(background:Image.Image,source:Image.Image, *, zoom:float,
           cx:float,cy:float,depth:float,role:str)->Image.Image:
    w,h=background.size
    base_zoom=1.06 if role=="background" else 1.0
    factor=max(1.0,zoom*base_zoom)
    zw,zh=math.ceil(w*factor),math.ceil(h*factor)
    if role=="background":
        layer=_cover_rgba(source,(zw,zh))
    else:
        layer=_cover_rgba(source,(zw,zh))
    extra_x=zw-w
    extra_y=zh-h
    # Background moves least, foreground most; normalized pan stays inside canvas.
    travel=0.18+0.42*(depth+1)/2
    x=round(extra_x/2+cx*extra_x*travel)
    y=round(extra_y/2+cy*extra_y*travel)
    x=max(0,min(extra_x,x))
    y=max(0,min(extra_y,y))
    return layer.crop((x,y,x+w,y+h))


def _blend(dst:Image.Image, layer:Image.Image, mode:str, opacity:float)->Image.Image:
    if opacity<=0:
        return dst
    layer=layer.copy()
    if opacity<1:
        layer.putalpha(layer.getchannel("A").point(lambda a:int(a*opacity)))
    if mode=="over":
        return Image.alpha_composite(dst,layer)
    opaque=dst.convert("RGB")
    light=layer.convert("RGB")
    if mode=="screen":
        result=ImageChops.screen(opaque,light)
    elif mode=="add":
        result=ImageChops.add(opaque,light,scale=1.0,offset=0)
    else:
        raise ValueError("Unsupported blend mode")
    return Image.composite(result,opaque,layer.getchannel("A")).convert("RGBA")


class CinematicPanel:
    """Cache source layers once per scene; composite a requested frame."""
    def __init__(self, scene:dict, assets_root:Path, width:int,height:int):
        validate_premium_scene(scene,assets_root,publish=False,min_width=width,min_height=height)
        self.scene=scene
        self.width,self.height=width,height
        self.layers=[]
        for layer in scene["cinematic"]["layers"]:
            path=resolve_asset(assets_root,layer["path"])
            with Image.open(path) as image:
                image.load()
                blink_overlay=None
                if layer.get("motion"):
                    validate_motion(layer["motion"],assets_root,publish=False)
                    blink_spec=layer["motion"].get("blink")
                    if blink_spec:
                        with Image.open(resolve_asset(assets_root,blink_spec["path"])) as overlay:
                            overlay.load()
                            blink_overlay=overlay.convert("RGBA")
                self.layers.append((layer,image.convert("RGBA"),blink_overlay))
        self.grain=float(scene["cinematic"].get("grain",0.0) or 0.0)
        self.vignette=float(scene["cinematic"].get("vignette",0.14) or 0.0)
        if not 0<=self.vignette<=0.85:
            raise ValueError("vignette must be 0..0.85")
        if not 0<=self.grain<=0.25:
            raise ValueError("grain must be 0..0.25")
        self.mask=self._vignette_mask()
        # Fixed sequence prevents stochastic temporal shimmer (no random noise each frame).
        rng=np.random.default_rng(32018)
        self.noise=(rng.standard_normal((height,width))*5).astype(np.float32) if self.grain else None

    def _vignette_mask(self)->Image.Image:
        h,w=self.height,self.width
        ys=np.linspace(-1,1,h,dtype=np.float32)[:,None]
        xs=np.linspace(-1,1,w,dtype=np.float32)[None,:]
        radial=np.clip((xs*xs*.6+ys*ys*.7),0,1)
        brightness=(1-self.vignette*radial)*255
        return Image.fromarray(np.uint8(brightness).repeat(w,axis=1) if brightness.shape[1]==1
                               else np.uint8(brightness),"L")

    def frame(self,t:float)->Image.Image:
        x,y,zoom=camera_state(self.scene,t)
        canvas=Image.new("RGBA",(self.width,self.height),(0,0,0,255))
        for meta,im,blink_overlay in self.layers:
            depth=float(meta.get("depth",0))
            if meta.get("motion"):
                moving=animate_actor(im,meta["motion"],t,overlay=blink_overlay)
            else:
                moving=im
            current=_place(canvas,moving,zoom=zoom,cx=x,cy=y,depth=depth,
                           role=meta["role"])
            canvas=_blend(canvas,current,meta.get("blend","over"),
                          float(meta.get("opacity",1)))
        image=canvas.convert("RGB")
        if self.vignette>0:
            black=Image.new("RGB",image.size,(0,0,0))
            image=Image.composite(image,black,self.mask)
        if self.grain and self.noise is not None:
            data=np.asarray(image,dtype=np.float32)
            data=np.uint8(np.clip(data+self.noise[:,:,None]*(self.grain*6),0,255))
            image=Image.fromarray(data,"RGB")
        return image
