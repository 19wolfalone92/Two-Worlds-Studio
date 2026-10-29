"""Deterministic 2.5D point-light grade for the existing Studio compositor.

A stylized screen-space light pass, NOT physically correct 3D relighting.
"""
from __future__ import annotations
import math
import numpy as np
from PIL import Image


def validate_lights(lights):
    if lights is None:
        return []
    if not isinstance(lights,list) or len(lights)>4:
        raise ValueError('Use at most four authored screen-space lights')
    for light in lights:
        if not isinstance(light,dict) or set(light)-{
            'x','y','radius','rgb','from','to','flicker_hz'}:
            raise ValueError('Invalid light parameters')
        for k,lo,hi in [('x',0,1),('y',0,1),('radius',.05,1.5),
                        ('from',-.4,.7),('to',-.4,.7),('flicker_hz',0,8)]:
            default=0 if k in ('from','to','flicker_hz') else None
            val=light.get(k,default)
            if (type(val) not in (float,int) or not math.isfinite(val) or
                    not lo<=val<=hi):
                raise ValueError(f'Light {k} out of range')
        rgb=light.get('rgb')
        if (not isinstance(rgb,list) or len(rgb)!=3 or any(
                type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=255
                for v in rgb)):
            raise ValueError('Light RGB should be three 0..255 values')
    return lights


class LightStage:
    def __init__(self,lights,size):
        self.lights=validate_lights(lights)
        self.size=size
        w,h=size
        yy,xx=np.ogrid[:h,:w]
        self.falloffs=[np.exp(-2*((xx-w*l['x'])**2+(yy-h*l['y'])**2)/
                              (min(w,h)*l['radius'])**2).astype(np.float32)
                       for l in self.lights]

    def frame(self,img:Image.Image,t:float)->Image.Image:
        if not 0<=t<=1 or not math.isfinite(t):
            raise ValueError('Invalid light timeline')
        if img.size!=self.size:
            raise ValueError('Light layer and scene dimensions disagree')
        if not self.lights:return img
        pixels=np.asarray(img.convert('RGB'),dtype=np.float32).copy()
        eased=t*t*(3-2*t)
        for light,falloff in zip(self.lights,self.falloffs):
            value=(1-eased)*light.get('from',0)+eased*light.get('to',0)
            hz=light.get('flicker_hz',0)
            if hz:value*=1+.04*math.sin(t*hz*2*math.pi)
            rgb=np.array(light['rgb'],dtype=np.float32)
            if value<0:
                pixels*=1+value*falloff[:,:,None]
            else:
                tint=rgb/255
                # Screen-style grade, preserves texture and stays deterministic.
                pixels+=value*falloff[:,:,None]*(255-pixels)*tint[None,None,:]
        return Image.fromarray(np.uint8(np.clip(pixels,0,255)),'RGB')
