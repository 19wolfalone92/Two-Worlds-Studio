"""Expressive facial sprites attached to an articulated Studio head bone.

Use artist-drawn aligned lid/viseme plates that REPLACE underlying eye/mouth
pixels with opaque repaint, never stack a closed lid over an open eyeball.
Timed visemes take precedence; amplitude fallback is explicitly non-phonemic.
"""
from __future__ import annotations
from bisect import bisect_right
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image

from studio.character_engine import CharacterRig, _transformed_patch
from studio.lipsync import PhonemeMotion, REST
from studio.premium_assets import resolve_asset


def _plate(root:Path,spec:dict,size:tuple[int,int], *, master=False):
    if not isinstance(spec,dict) or not isinstance(spec.get('sha256'),str):
        raise ValueError('Every face plate requires a pinned PNG hash')
    file=resolve_asset(root,spec.get('path'))
    if hashlib.sha256(file.read_bytes()).hexdigest()!=spec['sha256']:
        raise ValueError('Face plate SHA-256 mismatch')
    if master:
        from studio.premium_assets import LICENSES
        if (spec.get("review_status") != "approved" or spec.get("rights") not in LICENSES
                or len(str(spec.get("rights_note", "")).strip()) < 8):
            raise ValueError("Master face plate needs review and documented rights")
    with Image.open(file) as source:
        source.load()
        if source.size!=size or 'A' not in source.getbands():
            raise ValueError('Face plates must have actor-canvas RGBA coordinates')
        rgba=source.convert('RGBA')
    bounds=rgba.getchannel('A').getbbox()
    if bounds is None:
        raise ValueError('Empty facial plate')
    return rgba.crop(bounds),bounds


def _ease_field(value: np.ndarray) -> np.ndarray:
    unit = np.clip(value, 0., 1.)
    return unit * unit * (3. - 2. * unit)


def _sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Subpixel sampling of the original lip texture."""
    h, w = image.shape[:2]
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    left, top = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    right, bottom = np.minimum(left + 1, w - 1), np.minimum(top + 1, h - 1)
    fx, fy = (x - left)[..., None], (y - top)[..., None]
    upper = image[top, left] * (1 - fx) + image[top, right] * fx
    lower = image[bottom, left] * (1 - fx) + image[bottom, right] * fx
    return upper * (1 - fy) + lower * fy


class MouthMesh:
    """Articulate the registered lip seam and jaw on the source head sprite."""

    def __init__(self, rig: CharacterRig, spec: dict):
        if not isinstance(spec, dict) or set(spec) - {"seam", "max_open"}:
            raise ValueError("Mouth mesh requires a registered lip seam")
        points = spec.get("seam")
        if (not isinstance(points, list) or not 3 <= len(points) <= 12 or any(
                not isinstance(pair, list) or len(pair) != 2 or any(
                    type(v) not in (int, float) or not 0 <= v <= 1 for v in pair)
                for pair in points)):
            raise ValueError("Mouth seam must contain 3–12 normalized points")
        max_open = spec.get("max_open", .20)
        if type(max_open) not in (int, float) or not .08 <= max_open <= .30:
            raise ValueError("Mouth opening must be within .08–.30 of lip width")
        self.head, bounds = rig.parts["head"][1]
        coords = np.asarray(points, dtype=np.float32) * rig.size - bounds[:2]
        xs, ys = coords.T
        width = float(xs[-1] - xs[0])
        if (not 8 <= width <= min(self.head.width * .8, 650) or np.any(np.diff(xs) < 1)
                or float(np.ptp(ys)) > width * .55
                or any(not (0 <= x < self.head.width and 0 <= y < self.head.height)
                       for x, y in coords)):
            raise ValueError("Mouth seam must be ordered and lie inside the head sprite")
        if any(self.head.getpixel((int(x), int(y)))[3] < 224 for x, y in coords):
            raise ValueError("Mouth seam must lie on opaque facial artwork")
        left = max(0, math.floor(xs[0] - .36 * width))
        right = min(self.head.width, math.ceil(xs[-1] + .36 * width))
        top = max(0, math.floor(min(ys) - .40 * width))
        bottom = min(self.head.height, math.ceil(max(ys) + .78 * width))
        self.box = left, top, right, bottom
        self.source = np.asarray(self.head.crop(self.box), dtype=np.float32)
        self.xs, self.ys = xs - left, ys - top
        self.center = float((self.xs[0] + self.xs[-1]) / 2)
        self.width, self.max_open = width, max_open
        self.y, self.x = np.indices(self.source.shape[:2], dtype=np.float32)
        self.seam = np.interp(self.x, self.xs, self.ys)
        self.lip_reach = _ease_field(1 - np.abs(self.y - self.seam) / (.35 * width))
        # Keep the face contour and nearby hair fixed in a three-quarter view.
        self.side_reach = _ease_field((.60 - np.abs(self.x - self.center) / width) / .18)

    def frame(self, pose: tuple[float, float, float]) -> Image.Image:
        if pose == REST:
            return self.head
        opening, spread, rounding = pose
        width = self.width
        horizontal = spread / (1 + spread)
        source_x = self.x - (self.x - self.center) * horizontal * self.lip_reach * self.side_reach
        centerline = np.interp(source_x, self.xs, self.ys)
        distance = 2 * (source_x - self.center) / width
        shape = np.maximum(0., 1. - distance ** 2) ** (.85 + .78 * rounding)
        gap = width * self.max_open * opening * shape
        upper = centerline - gap * (.18 + .10 * rounding)
        lower = upper + gap
        above = _ease_field(1. - (upper - self.y) / (.31 * width))
        below = _ease_field(1. - (self.y - lower) / (.56 * width))
        # Keep the original lip edges and displace the neighboring skin
        # instead of layering independently drawn shapes onto a moving face.
        seam_ink = np.minimum(.8, .18 * gap)
        source_y = np.where(self.y <= upper,
                            self.y + (centerline - upper - seam_ink) * above,
                            self.y - (lower - centerline - seam_ink) * below)
        skin = _sample(self.source, source_x, source_y)
        valid_skin = _ease_field((np.minimum(skin[..., 3], self.source[..., 3]) - 210) / 44)
        rgb = self.source[..., :3] + (skin[..., :3] - self.source[..., :3]) * valid_skin[..., None]

        # Subpixel edge coverage and a shaded cavity. A subtle lower value
        # catches the light without drawing a separate set of teeth.
        cavity = np.clip(np.minimum(self.y + .5, lower) -
                         np.maximum(self.y - .5, upper), 0, 1) * valid_skin
        depth = np.clip((self.y - upper) / np.maximum(gap, 1), 0, 1)
        shade = np.maximum(0., (depth - .34) / .66) * np.sqrt(shape)
        dark = np.array([32., 16., 16.])
        inner = dark + shade[..., None] * np.array([48., 21., 17.])
        rgb = rgb * (1. - cavity[..., None]) + inner * cavity[..., None]
        pixels = np.dstack((rgb, self.source[..., 3]))
        head = self.head.copy()
        head.paste(Image.fromarray(np.clip(pixels + .5, 0, 255).astype(np.uint8)), self.box[:2])
        return head


class PerformanceRig:
    """Re-render eyes/lips on head parent transform at each frame."""
    def __init__(self,rig:CharacterRig,spec:dict,root:Path,*,
                 duration:float,voice_levels:list[float]|None=None,
                 voice_step_ms:int=50,voice_delay_ms:int=0,
                 activity:dict|None=None, phonemes:PhonemeMotion|None=None, master=False):
        if rig.config.get('camera_view')=='back':
            raise ValueError('Face overlays cannot be attached to the BACK of a head')
        if 'head' not in rig.parts:
            raise ValueError('Facial rig needs the head bone')
        if not math.isfinite(duration) or duration<=0:
            raise ValueError('Positive scene duration required')
        self.rig=rig;self.duration=duration
        self.voice_levels=voice_levels
        self.activity=activity
        self.phonemes=phonemes
        self.voice_step_ms=voice_step_ms;self.voice_delay_ms=voice_delay_ms
        if any(not isinstance(spec.get(key, {}), dict) for key in ("eyes", "mouth")):
            raise ValueError("Facial plates must be keyed by shape")
        self.eyes={key:_plate(root,plate,rig.size,master=master)
                   for key,plate in spec.get('eyes',{}).items()}
        self.mouth={key:_plate(root,plate,rig.size,master=master)
                    for key,plate in spec.get('mouth',{}).items()}
        mesh = spec.get("mouth_mesh")
        if mesh is not None and (self.mouth or spec.get("visemes") or phonemes is None):
            raise ValueError("Mouth mesh needs a single phoneme timing source")
        self.mesh = MouthMesh(rig, mesh) if mesh is not None else None
        if self.eyes and not {'open','half','closed'}<=set(self.eyes):
            raise ValueError('Blink requires open, half and fully closed eye plates')
        if self.mouth and 'rest' not in self.mouth:
            raise ValueError('Mouth needs neutral rest plate')
        if not self.eyes and not self.mouth and self.mesh is None:
            raise ValueError('At least one genuine facial performance layer required')
        self.blinks=spec.get('blinks',[])
        if not isinstance(self.blinks,list) or any(
            type(t) not in (int,float) or not math.isfinite(t) or not 0<=t<duration
            for t in self.blinks):
            raise ValueError('Invalid blink timestamps')
        self.blink_seconds=float(spec.get('blink_seconds',.18))
        if not .08<=self.blink_seconds<=.5:
            raise ValueError('Unnatural blink length')
        self.cues=spec.get('visemes',[])
        if not isinstance(self.cues,list):
            raise ValueError('Visemes must be a time-ordered list')
        prev=-1
        for item in self.cues:
            if (not isinstance(item,dict) or set(item)!={'at','shape'} or
                type(item['at']) not in (float,int) or
                not math.isfinite(item['at']) or not 0<=item['at']<=duration or
                item['at']<=prev or not isinstance(item['shape'], str)
                or item['shape'] not in self.mouth):
                raise ValueError('Viseme cue is invalid or cannot be rendered')
            prev=item['at']
        self.cue_times = [cue["at"] for cue in self.cues]
        self.mode=('mfa_phoneme_mesh' if self.mesh is not None else
                   'waveform_gated_visemes' if self.cues and activity is not None else
                   'timed_visemes' if self.cues else
                   'amplitude_only_not_phonemes' if voice_levels is not None else
                   'silent')

    def _eyes(self,sec:float):
        if not self.eyes:
            return None
        for at in self.blinks:
            elapsed=sec-at
            if 0<=elapsed<self.blink_seconds:
                phase=elapsed/self.blink_seconds
                return 'closed' if .30<=phase<.7 else 'half'
        return 'open'

    def _mouth(self,sec:float):
        if not self.mouth:
            return None
        if self.cues:
            time = sec - self.voice_delay_ms / 1000
            if self.activity is not None:
                i = math.floor(time * 1000 / self.activity["step_ms"] + 1e-9)
                active = self.activity["active"]
                if not 0 <= i < len(active) or not active[i]:
                    return "rest"
            index = bisect_right(self.cue_times, time + 1e-9) - 1
            return self.cues[index]["shape"] if index >= 0 else "rest"
        if self.voice_levels is None:
            return 'rest'
        idx=round((sec*1000-self.voice_delay_ms)/self.voice_step_ms)
        if not 0<=idx<len(self.voice_levels):
            return 'rest'
        amp=self.voice_levels[idx]
        if amp<.22:return 'rest'
        return 'wide' if amp>=.6 and 'wide' in self.mouth else (
            'small' if 'small' in self.mouth else 'rest')

    def _mouth_plate(self, seconds):
        shape = self._mouth(seconds)
        current, bounds = self.mouth[shape]
        if shape == "rest" or not self.cues:
            return current, bounds
        time = seconds - self.voice_delay_ms / 1000
        index = bisect_right(self.cue_times, time + 1e-9) - 1
        blend = min(1., max(0., (time - self.cue_times[index]) / .045))
        # Use the drawn neutral pose immediately in silence. Within speech,
        # crossfade registered repaint plates across brief shape transitions.
        previous, prev_bounds = self.mouth[self.cues[index - 1]["shape"] if index else "rest"]
        if blend < 1 and bounds == prev_bounds:
            return Image.blend(previous, current, blend), bounds
        return current, bounds

    def frame(self,t:float, *, seconds:float|None=None)->Image.Image:
        if not isinstance(t,(int,float)) or not math.isfinite(t) or not 0<=t<=1:
            raise ValueError('Normalized finite animation time required')
        sec=t*self.duration if seconds is None else seconds
        if self.mesh is not None:
            pose = self.phonemes.pose(sec - self.voice_delay_ms / 1000)
            frame = self.rig.frame(t, head=self.mesh.frame(pose))
        else:
            frame = self.rig.frame(t)
        if not (self.eyes or self.mouth):
            return frame
        head=self.rig.pose(t)['head']
        if self.eyes:
            _transformed_patch(frame,self.eyes[self._eyes(sec)],head)
        if self.mouth:
            _transformed_patch(frame,self._mouth_plate(sec),head)
        return frame
