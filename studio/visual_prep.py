"""Generate an original layered rendering *smoke-test*, not a narrative episode.

Uses vector geometry & noise to verify alpha, multilayer depth, camera,
lighting and export without any paid illustration service or GPU.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from studio.render import render


def make_demo_assets(directory:Path,width:int=720,height:int=1280)->tuple[Path,dict]:
    from studio.premium_assets import inspect_image
    images=directory/"assets"/"twilight"
    images.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261028)
    ys=np.linspace(0,1,height,dtype=np.float32)[:,None,None]
    xs=np.linspace(0,1,width,dtype=np.float32)[None,:,None]
    top=np.asarray([22,16,46],dtype=np.float32)[None,None,:]
    bottom=np.asarray([112,50,80],dtype=np.float32)[None,None,:]
    colors=(1-ys)*top+ys*bottom
    colors=np.broadcast_to(colors,(height,width,3)).copy()
    colors+=(rng.standard_normal((height,width,1))*2.0).astype(np.float32)
    colors+=np.sin(xs*5+ys*9)*3
    bg=Image.fromarray(np.uint8(np.clip(colors,0,255)),"RGB").convert("RGBA")
    d=ImageDraw.Draw(bg,"RGBA")
    for x,y,r,color in [(round(width*.68),round(height*.19),72,(227,228,240,255)),
                         (round(width*.28),round(height*.21),46,(219,143,176,255))]:
        for k in range(6,0,-1):
            rr=r+17*k
            d.ellipse((x-rr,y-rr,x+rr,y+rr),fill=color[:3]+(max(1,round(4*(7-k))),))
        d.ellipse((x-r,y-r,x+r,y+r),fill=color)
        d.arc((x-r,y-r,x+r,y+r),90,300,fill=(79,56,89,180),width=5)
    for i in range(110):
        x=int(rng.integers(0,width))
        y=int(rng.integers(0,height*.6))
        rad=1 if i%6 else 2
        d.ellipse((x-rad,y-rad,x+rad,y+rad),fill=(242,206,226,90+i%3*45))
    # Far hills and citadel silhouette, kept on background.
    hills=[(0,height*.7)]+[(x,round(height*(.63+.04*math.sin(x/83)))) for x in range(0,width+35,35)]
    d.polygon(hills+[(width,height),(0,height)],fill=(38,29,55,255))
    bg_path=images/"background.png";bg.save(bg_path)

    mid=Image.new("RGBA",(width,height),(0,0,0,0))
    m=ImageDraw.Draw(mid,"RGBA")
    base_y=round(height*.78)
    # Layer of separate towers, parallax silhouettes and highlights.
    for i in range(12):
        tx=int(i*68-25)
        tw=38+(i*3)%22
        ht=190+((i*61)%235)
        ytop=base_y-ht
        color=(23+i%3*4,17+i%4*3,40+i%3*5,253)
        m.rectangle((tx,ytop,tx+tw,base_y+160),fill=color)
        m.polygon((tx-10,ytop,tx+tw//2,ytop-65,tx+tw+10,ytop),fill=(32,20,49,255))
        for j in range(3):
            yy=ytop+34+j*48
            if yy<base_y:
                m.polygon((tx+tw//2-8,yy+15,tx+tw//2-8,yy,
                           tx+tw//2,yy-16,tx+tw//2+8,yy,tx+tw//2+8,yy+15),
                          fill=(216,100+j*14,126,205))
        m.line((tx,ytop,tx,base_y),fill=(147,85,111,130),width=2)
    m.polygon([(0,base_y),(width,base_y-35),(width,height),(0,height)],
              fill=(26,21,41,255))
    mid_path=images/"midground.png";mid.save(mid_path)

    front=Image.new("RGBA",(width,height),(0,0,0,0))
    f=ImageDraw.Draw(front,"RGBA")
    # An ancient arch frames the background. Large foreground in alpha.
    left=round(width*.12);right=round(width*.86)
    f.rectangle((0,0,left,height),fill=(11,14,26,245))
    f.rectangle((right,0,width,height),fill=(11,14,26,245))
    f.rectangle((0,0,width,115),fill=(13,12,27,255))
    for i in range(14):
        x=14+i*52
        y=86+round(math.cos(i*.3)*7)
        f.line((x,y,x+12,height),fill=(122,63,91,65),width=2)
    for i in range(11):
        px=int((i*83)%width)
        py=int(height*(.76+.02*(i%3)))
        f.ellipse((px-26,py-8,px+26,py+10),fill=(193,113,150,26))
    front=front.filter(ImageFilter.GaussianBlur(1.6))
    front_path=images/"foreground.png";front.save(front_path)
    layers=[]
    for role,p,depth in [("background",bg_path,-0.85),("midground",mid_path,0.15),("foreground",front_path,0.90)]:
        layers.append({"role":role,"path":str(p.relative_to(directory/"assets")),
                       "depth":depth,"opacity":1.0,"blend":"over",
                       "sha256":inspect_image(p)["sha256"],
                       "rights":"original","rights_note":"Procedural original vector art, technical demo only",
                       "review_status":"unreviewed"})
    scene={"id":"S01","duration":3.5,"palette":"noctaris","effect":"embers",
           "heading":"TWO WORLDS","caption":"2.5D layered visual pipeline | TECHNICAL DEMO",
           "cinematic":{"layers":layers,"camera":{"start":{"x":-.30,"y":-.10,"zoom":1.04},
             "end":{"x":.30,"y":.10,"zoom":1.20}},"grain":.06,"vignette":.25}}
    episode={"id":"E99","title":"NOT A STORY EPISODE — VISUAL TEST","fps":10,
             "scenes":[scene]}
    spec=directory/"visual_demo.json"
    spec.write_text(json.dumps(episode,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return spec,episode


def main():
    p=argparse.ArgumentParser(description="Build illustrative layered art benchmark")
    p.add_argument("--demo",action="store_true",help="Explicit consent to create the procedural demo")
    p.add_argument("--output-dir",type=Path,default=Path("dist/premium_demo"))
    p.add_argument("--no-video",action="store_true")
    p.add_argument("--width",type=int,default=360)
    p.add_argument("--height",type=int,default=640)
    p.add_argument("--fps",type=int,default=10)
    args=p.parse_args()
    if not args.demo:
        p.error("Only --demo is supported; no scripts or AI image generation occur")
    path,episode=make_demo_assets(args.output_dir)
    if not args.no_video:
        video=args.output_dir/"VISUAL_TECH_DEMO.mp4"
        result=render(path,video,args.output_dir/"assets",args.width,args.height,args.fps)
        print(json.dumps(result))
    (args.output_dir/"README.txt").write_text(
        "Visual pipeline demo; all procedural drawings are illustrative placeholders. "
        "No AI-generated professional art, story content or TikTok publication. "
        "See docs/WORK_VISUAL_HANDOFF.md for production layered assets.\n",encoding="utf-8")


if __name__=="__main__":
    main()
