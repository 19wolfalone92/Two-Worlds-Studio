"""Conservative automatic matte extraction for GPU character sheets.

Accepts a nearly uniform background and one central upright silhouette.
Marks the result UNREVIEWED: hair and cloth edges may need corrections.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
try:
    import cv2
except ImportError:
    cv2=None


def matte_plain_background(source:Path,output:Path)->dict:
    if cv2 is None:
        raise RuntimeError('Background matting requires OpenCV')
    if output.exists():
        raise FileExistsError('Never overwrite pre-existing character mattes')
    with Image.open(source) as im:
        im.load()
        if im.format!='PNG':
            raise ValueError('Expect an original generated PNG')
        rgb=np.array(im.convert('RGB'),dtype=np.uint8)
    h,w=rgb.shape[:2]
    if w<128 or h<192 or w*h>12_000_000:
        raise ValueError('Invalid full-body source dimensions')
    k=max(3,round(min(w,h)*.03))
    edge=np.concatenate((rgb[:k].reshape(-1,3),rgb[-k:].reshape(-1,3),
                         rgb[:,:k].reshape(-1,3),rgb[:,-k:].reshape(-1,3)),axis=0)
    bg=np.median(edge,axis=0)
    variation=float(np.percentile(np.linalg.norm(edge.astype(np.float32)-bg,axis=1),90))
    if variation>35:
        raise ValueError('Background too complex for safe automatic alpha extraction')
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab=cv2.cvtColor(np.uint8([[bg]]),cv2.COLOR_RGB2LAB)[0,0].astype(np.float32)
    distance=np.sqrt(np.sum((lab-bg_lab)**2,axis=2))
    threshold=max(23,variation*1.4)
    binary=np.uint8(distance>threshold)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel,iterations=2)
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(binary,8)
    if count<=1:
        raise ValueError('No foreground subject distinguishable from background')
    component=max(range(1,count),key=lambda i:stats[i,cv2.CC_STAT_AREA])
    x,y,bw,bh,area=map(int,stats[component])
    if not (.14<area/(w*h)<.75 and .54<bh/h<.98 and .18<bw/bh<.85 and
            x>2 and y>2 and x+bw<w-2 and y+bh<h-2):
        raise ValueError('Cannot safely isolate one full standing figure')
    mask=np.uint8(labels==component)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=1)
    mask=cv2.GaussianBlur(mask,(3,3),.65)
    rgba=np.dstack((rgb,mask))
    output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(rgba,'RGBA').save(output)
    return {'method':'conservative_uniform_background',
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'matte_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'source_background_p90_delta':round(variation,2),
            'area_ratio':round(area/(w*h),4),
            'owner_approved':False,'needs_edge_review':True}
