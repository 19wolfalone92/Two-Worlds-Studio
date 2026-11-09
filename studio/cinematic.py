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
    from studio.audio_master import audio_asset
    from studio.lipsync import voice_envelope, mouth_open
except ModuleNotFoundError:
    from premium_assets import resolve_asset, validate_premium_scene
    from actor_motion import animate_actor, validate_motion
    from audio_master import audio_asset
    from lipsync import voice_envelope, mouth_open


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


def layer_transform(size, source_size, camera, depth, role):
    """Single continuous inverse transform from screen pixels to the layer."""
    w, h = size
    sw, sh = source_size
    cx, cy, zoom = camera
    zoom *= 1.06 if role == "background" else 1.0
    fit = max(w / sw, h / sh)
    step = 1 / (fit * zoom)
    travel = 0.18 + 0.42 * (depth + 1) / 2
    # Use continuous overscan for the entire path, including changes in scale.
    x = (sw - w * step) / 2 + cx * travel * (w / fit - w * step)
    y = (sh - h * step) / 2 + cy * travel * (h / fit - h * step)
    x = min(sw - w * step, max(0., x))
    y = min(sh - h * step, max(0., y))
    return step, 0., max(0., x), 0., step, max(0., y)


def _place(background: Image.Image, source: Image.Image, *,
           zoom: float, cx: float, cy: float, depth: float, role: str) -> Image.Image:
    transform = layer_transform(background.size, source.size, (cx, cy, zoom), depth, role)
    return source.transform(background.size, Image.Transform.AFFINE, transform,
                            resample=Image.Resampling.BICUBIC)


def _blend(dst:Image.Image, layer:Image.Image, mode:str, opacity:float)->Image.Image:
    if opacity<=0:
        return dst
    if opacity<1:
        layer=layer.copy()
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
    def __init__(self, scene:dict, assets_root:Path, width:int,height:int, *, master=False):
        from studio.character_engine import CharacterRig
        from studio.cinematic_lighting import LightStage
        from studio.facial_actor import PerformanceRig
        from studio.lipsync import load_viseme_track, speech_activity

        validate_premium_scene(scene, assets_root, publish=master, min_width=width, min_height=height)
        self.scene=scene
        self.width,self.height=width,height
        self.layers=[]
        self.voice_levels=None
        self.voice_delay_ms=0
        self.voice_step_ms=50
        self.performance_modes = []
        self.active_actors = 0
        voice=scene.get("voice")
        layers = scene["cinematic"]["layers"]
        if sum(bool(x.get("facial", {}).get("viseme_track") or
                    x.get("facial", {}).get("visemes")) for x in layers) > 1 and voice:
            raise ValueError("One shot voice can drive only one speaking facial rig")
        clip = None
        if voice and any(x.get("motion", {}).get("mouth") or
                         x.get("facial", {}).get("mouth") for x in layers):
            clip = audio_asset(assets_root, voice.get("path"))
            self.voice_delay_ms = int(voice.get("offset_ms", 100))
        if clip is not None and any(x.get("motion", {}).get("mouth") for x in layers):
            self.voice_levels=voice_envelope(clip,step_ms=self.voice_step_ms)
        for layer in layers:
            path=resolve_asset(assets_root,layer["path"])
            with Image.open(path) as image:
                image.load()
                blink_overlay=None
                mouth_overlay=None
                if layer.get("motion"):
                    validate_motion(layer["motion"],assets_root,publish=False)
                    mouth_spec=layer["motion"].get("mouth")
                    if mouth_spec:
                        if self.voice_levels is None:
                            raise ValueError("Mouth animation requires a recorded scene voice")
                        with Image.open(resolve_asset(assets_root,mouth_spec["path"])) as lip:
                            lip.load()
                            mouth_overlay=lip.convert("RGBA")
                    blink_spec=layer["motion"].get("blink")
                    if blink_spec:
                        with Image.open(resolve_asset(assets_root,blink_spec["path"])) as overlay:
                            overlay.load()
                            blink_overlay=overlay.convert("RGBA")
                layer_image = ImageOps.exif_transpose(image).convert("RGBA")
            actor = None
            if layer.get("rig") is not None:
                actor = CharacterRig(layer["rig"], assets_root, layer_image.size, publish=master)
                self.active_actors += int(actor.stats["is_acting"])
                if layer.get("facial") is not None:
                    face = layer["facial"]
                    activity = None
                    if face.get("viseme_track") is not None:
                        if clip is None or face.get("visemes"):
                            raise ValueError("Recorded visemes require one shot voice and one timing source")
                        face = dict(face, visemes=load_viseme_track(assets_root, face["viseme_track"], clip))
                    if clip is not None and face.get("mouth") and face.get("visemes"):
                        activity = speech_activity(clip)
                        if not any(activity["active"]):
                            raise ValueError("The lip-sync voice contains no detectable speech activity")
                    actor = PerformanceRig(actor, face, assets_root,
                        duration=float(scene["duration"]), voice_delay_ms=self.voice_delay_ms,
                        activity=activity, master=master)
                    self.performance_modes.append(actor.mode)
            elif layer.get("deform") is not None:
                from studio.deformable_actor import DeformableActor
                actor = DeformableActor(layer_image, layer["deform"])
                self.active_actors += 1
            self.layers.append((layer, layer_image, blink_overlay, mouth_overlay, actor))
        self.lighting = LightStage(scene["cinematic"].get("lights"), (width, height))
        self.regions = scene.get("protected_regions", [])
        for region in self.regions:
            index = region["layer"]
            if index >= len(self.layers):
                raise ValueError("Protected region references a missing layer")
            layer, image, *_, actor = self.layers[index]
            if actor is not None or "motion" in layer or layer.get("opacity", 1) < .95:
                raise ValueError("Protected region must be on a stable, visible layer")
            w, h = image.size
            x0, y0, x1, y1 = region["box"]
            roi = image.getchannel("A").crop((math.floor(x0 * w), math.floor(y0 * h),
                                               math.ceil(x1 * w), math.ceil(y1 * h)))
            if roi.getextrema()[0] < 200:
                raise ValueError("Protected region contains transparent source pixels")
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
        return Image.fromarray(np.uint8(brightness))

    def frame(self,t:float, *, seconds:float | None = None)->Image.Image:
        camera = camera_state(self.scene, t)
        cx, cy, zoom = camera
        size = self.width, self.height
        clock = t * self.scene["duration"] if seconds is None and self.regions else seconds
        visible_regions = [r for r in self.regions if r["start"] <= clock < r["end"]]
        covered_regions = []
        canvas=Image.new("RGBA",(self.width,self.height),(0,0,0,255))
        for index, (meta,im,blink_overlay,mouth_overlay,actor) in enumerate(self.layers):
            depth=float(meta.get("depth",0))
            if "facial" in meta:
                moving = actor.frame(t, seconds=seconds)
            elif actor is not None:
                moving = actor.frame(t)
            elif meta.get("motion"):
                mouth_enabled=False
                if mouth_overlay is not None and self.voice_levels is not None:
                    mouth_enabled=mouth_open(self.voice_levels,t*float(self.scene["duration"]),
                       step_ms=self.voice_step_ms,
                       delay_ms=self.voice_delay_ms,
                       threshold=float(meta["motion"]["mouth"].get("threshold",.22)))
                moving=animate_actor(im,meta["motion"],t,overlay=blink_overlay,
                                      mouth_overlay=mouth_overlay,mouth_is_open=mouth_enabled)
            else:
                moving=im
            current = _place(canvas, moving, zoom=zoom, cx=cx, cy=cy, depth=depth, role=meta["role"])
            opacity = float(meta.get("opacity", 1))
            if covered_regions and opacity > .25 and meta.get("blend", "over") == "over":
                alpha = current.getchannel("A")
                if any(alpha.crop(box).getextrema()[1] * opacity >= 64 for box in covered_regions):
                    raise ValueError("Another layer occludes a protected region")
            for region in visible_regions:
                if region["layer"] != index:
                    continue
                step, _, x, _, _, y = layer_transform(size, moving.size, camera, depth, meta["role"])
                x0, y0, x1, y1 = region["box"]
                left, right = ((v * moving.width - x) / step for v in (x0, x1))
                top, bottom = ((v * moving.height - y) / step for v in (y0, y1))
                if left < 0 or top < 0 or right > self.width or bottom > self.height:
                    raise ValueError("Camera crops a protected region during its visibility window")
                if any(start <= clock < end and top < y1 * self.height and bottom > y0 * self.height
                       for start, end, y0, y1 in self.scene.get("caption_windows", ())):
                    raise ValueError("Burned captions may cover a protected region")
                covered_regions.append((math.floor(left), math.floor(top),
                                        math.ceil(right), math.ceil(bottom)))
            canvas=_blend(canvas,current,meta.get("blend","over"),
                          opacity)
        image=self.lighting.frame(canvas.convert("RGB"), t)
        if self.vignette>0:
            black=Image.new("RGB",image.size,(0,0,0))
            image=Image.composite(image,black,self.mask)
        if self.grain and self.noise is not None:
            data=np.asarray(image,dtype=np.float32)
            data=np.uint8(np.clip(data+self.noise[:,:,None]*(self.grain*6),0,255))
            image=Image.fromarray(data,"RGB")
        return image
