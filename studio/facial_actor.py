"""Expressive facial sprites attached to an articulated Studio head bone.

Use artist-drawn aligned lid/viseme plates that REPLACE underlying eye/mouth
pixels with opaque repaint, never stack a closed lid over an open eyeball.
Timed visemes take precedence; amplitude fallback is explicitly non-phonemic.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
from PIL import Image

from studio.character_engine import CharacterRig, _transformed_patch
from studio.premium_assets import resolve_asset


def _plate(root:Path,spec:dict,size:tuple[int,int]):
    if not isinstance(spec,dict) or not isinstance(spec.get('sha256'),str):
        raise ValueError('Every face plate requires a pinned PNG hash')
    file=resolve_asset(root,spec.get('path'))
    if hashlib.sha256(file.read_bytes()).hexdigest()!=spec['sha256']:
        raise ValueError('Face plate SHA-256 mismatch')
    with Image.open(file) as source:
        source.load()
        if source.size!=size or 'A' not in source.getbands():
            raise ValueError('Face plates must have actor-canvas RGBA coordinates')
        rgba=source.convert('RGBA')
    bounds=rgba.getchannel('A').getbbox()
    if bounds is None:
        raise ValueError('Empty facial plate')
    return rgba.crop(bounds),bounds


class PerformanceRig:
    """Re-render eyes/lips on head parent transform at each frame."""
    def __init__(self,rig:CharacterRig,spec:dict,root:Path,*,
                 duration:float,voice_levels:list[float]|None=None,
                 voice_step_ms:int=50,voice_delay_ms:int=0):
        if rig.config.get('camera_view')=='back':
            raise ValueError('Face overlays cannot be attached to the BACK of a head')
        if 'head' not in rig.parts:
            raise ValueError('Facial rig needs the head bone')
        if not math.isfinite(duration) or duration<=0:
            raise ValueError('Positive scene duration required')
        self.rig=rig;self.duration=duration
        self.voice_levels=voice_levels
        self.voice_step_ms=voice_step_ms;self.voice_delay_ms=voice_delay_ms
        self.eyes={key:_plate(root,plate,rig.size) for key,plate in spec.get('eyes',{}).items()}
        self.mouth={key:_plate(root,plate,rig.size) for key,plate in spec.get('mouth',{}).items()}
        if self.eyes and not {'open','half','closed'}<=set(self.eyes):
            raise ValueError('Blink requires open, half and fully closed eye plates')
        if self.mouth and 'rest' not in self.mouth:
            raise ValueError('Mouth needs neutral rest plate')
        if not self.eyes and not self.mouth:
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
                item['at']<=prev or item['shape'] not in self.mouth):
                raise ValueError('Viseme cue is invalid or cannot be rendered')
            prev=item['at']
        self.mode=('timed_visemes' if self.cues else
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
            shape='rest'
            for cue in self.cues:
                if cue['at']>sec:break
                shape=cue['shape']
            return shape
        if self.voice_levels is None:
            return 'rest'
        idx=round((sec*1000-self.voice_delay_ms)/self.voice_step_ms)
        if not 0<=idx<len(self.voice_levels):
            return 'rest'
        amp=self.voice_levels[idx]
        if amp<.22:return 'rest'
        return 'wide' if amp>=.6 and 'wide' in self.mouth else (
            'small' if 'small' in self.mouth else 'rest')

    def frame(self,t:float)->Image.Image:
        if not isinstance(t,(int,float)) or not math.isfinite(t) or not 0<=t<=1:
            raise ValueError('Normalized finite animation time required')
        frame=self.rig.frame(t)
        head=self.rig.pose(t)['head']
        sec=t*self.duration
        if self.eyes:
            _transformed_patch(frame,self.eyes[self._eyes(sec)],head)
        if self.mouth:
            _transformed_patch(frame,self.mouth[self._mouth(sec)],head)
        return frame
