"""Continuous single-image deformation for TWO WORLDS Studio.

Every frame warps one continuous pixel field, not a rigid cutout collage.
Small acting motions ONLY; not 3D IK or neural video generation.
"""
from __future__ import annotations
import math
import numpy as np
from PIL import Image
try:
    import cv2
except ImportError:
    cv2=None


def validate_deform(spec):
    if not isinstance(spec,dict) or spec.get('schema')!=1:
        raise ValueError('Deformation requires schema=1')
    pts=spec.get('controls')
    if not isinstance(pts,list) or not 2<=len(pts)<=15:
        raise ValueError('Deformation requires 2..15 control points')
    ids=set();acting=False
    for ctrl in pts:
        if not isinstance(ctrl,dict) or not isinstance(ctrl.get('name'),str) or ctrl['name'] in ids:
            raise ValueError('Unique named controls required')
        ids.add(ctrl['name'])
        for name,lo,hi in [('x',0,1),('y',0,1),('radius',.025,.5)]:
            n=ctrl.get(name)
            if type(n) not in (int,float) or not math.isfinite(n) or not lo<=n<=hi:
                raise ValueError(f'Deform {name} is invalid')
        keys=ctrl.get('keys')
        if not isinstance(keys,list) or not 1<=len(keys)<=60:
            raise ValueError('Deform control requires authored keyframes')
        prev=-1
        for item in keys:
            if not isinstance(item,dict) or set(item)!={'at','dx','dy'}:
                raise ValueError('Deform keyframe requires at, dx, dy')
            for name,lo,hi in [('at',0,1),('dx',-.10,.10),('dy',-.10,.10)]:
                val=item[name]
                if type(val) not in (float,int) or not math.isfinite(val) or not lo<=val<=hi:
                    raise ValueError('Deform control goes beyond allowed safe movement')
            if item['at']<=prev:raise ValueError('Deform keyframes not increasing')
            prev=item['at']
        if any(k['dx']!=keys[0]['dx'] or k['dy']!=keys[0]['dy'] for k in keys[1:]):
            acting=True
    if not acting:raise ValueError('Static pixels are not animation')
    return True


def _interpolate(keys,t):
    if t<=keys[0]['at']:return keys[0]['dx'],keys[0]['dy']
    if t>=keys[-1]['at']:return keys[-1]['dx'],keys[-1]['dy']
    for a,b in zip(keys,keys[1:]):
        if a['at']<=t<=b['at']:
            u=(t-a['at'])/(b['at']-a['at']);u=u*u*(3-2*u)
            return a['dx']+(b['dx']-a['dx'])*u,a['dy']+(b['dy']-a['dy'])*u
    raise AssertionError('unreachable')


class DeformableActor:
    def __init__(self,image:Image.Image,config:dict):
        if cv2 is None:
            raise RuntimeError('OpenCV needed for deformable acting; install requirements-cinematic.txt')
        validate_deform(config)
        self.spec=config
        self.actor=np.asarray(image.convert('RGBA'),dtype=np.uint8)
        self.w,self.h=image.size
        yy,xx=np.mgrid[:self.h,:self.w].astype(np.float32)
        self.xx=xx;self.yy=yy
        self.influences=[]
        for ctrl in config['controls']:
            r=min(self.w,self.h)*ctrl['radius']
            d=((xx-ctrl['x']*self.w)**2+(yy-ctrl['y']*self.h)**2)/(2*r*r)
            self.influences.append(np.exp(-np.minimum(d,100)).astype(np.float32))

    def frame(self,t:float)->Image.Image:
        if type(t) not in (int,float) or not math.isfinite(t) or not 0<=t<=1:
            raise ValueError('Invalid deformation time')
        dx=np.zeros_like(self.xx);dy=np.zeros_like(self.yy)
        weight=np.ones_like(self.xx,dtype=np.float32)*.1
        for ctrl,falloff in zip(self.spec['controls'],self.influences):
            shift_x,shift_y=_interpolate(ctrl['keys'],t)
            dx+=falloff*shift_x*self.w
            dy+=falloff*shift_y*self.h
            weight+=falloff
        x=(self.xx-dx/weight).astype(np.float32)
        y=(self.yy-dy/weight).astype(np.float32)
        result=cv2.remap(self.actor,x,y,cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
        return Image.fromarray(result,'RGBA')
