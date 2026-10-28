"""Make scene contact sheets for an illustrator and a human QC reviewer.

The labels say PREVIEW or READY-FOR-HUMAN-REVIEW, not verified likeness.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from studio.premium_assets import review_episode
from studio.cinematic import CinematicPanel
from studio.comic_art import panel as standin_panel
from studio.render import gradient

def font(sz:int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(p).is_file():
            return ImageFont.truetype(p,sz)
    return ImageFont.load_default()

def contact_sheet(episode:dict, asset_root:Path, output:Path,
                  *, thumb_width:int=216,thumb_height:int=384,cols:int=3)->dict:
    if cols<1 or cols>6:
        raise ValueError("cols must be 1..6")
    info=review_episode(episode,asset_root,publish=False,
                        min_width=thumb_width,min_height=thumb_height)
    scenes=episode["scenes"]
    gap=16;heading_h=42;foot_h=76
    rows=math.ceil(len(scenes)/cols)
    w=cols*thumb_width+(cols+1)*gap
    h=rows*(thumb_height+heading_h+foot_h)+(rows+1)*gap+56
    out=Image.new("RGB",(w,h),(13,19,31))
    d=ImageDraw.Draw(out)
    title=str(episode.get("id","SCENES"))+" — REVIEW BOARD (NOT APPROVED)"
    d.text((gap,12),title,fill=(234,233,230),font=font(19))
    y0=56+gap
    status=[]
    for i,scene in enumerate(scenes):
        col=i%cols;row=i//cols
        x=gap+col*(thumb_width+gap);y=y0+row*(thumb_height+heading_h+foot_h+gap)
        if scene.get("cinematic"):
            art=CinematicPanel(scene,asset_root,thumb_width,thumb_height).frame(.5)
            mode="LAYERED PREVIEW"
        elif scene.get("background"):
            with Image.open(asset_root/scene["background"]) as source:
                art=ImageOps.fit(source.convert("RGB"),(thumb_width,thumb_height))
            mode="SINGLE IMAGE"
        elif scene.get("art"):
            art=standin_panel(scene,thumb_width,thumb_height)
            mode="PROCEDURAL STAND-IN"
        else:
            art=gradient((thumb_width,thumb_height),scene.get("palette","earth"))
            mode="COLOR STAND-IN"
        out.paste(art,(x,y+heading_h))
        d.text((x,y+4),str(scene["id"])+": "+str(scene.get("heading",""))[:22],
               fill=(237,209,183),font=font(14))
        foot_y=y+heading_h+thumb_height+9
        d.text((x,foot_y),mode,fill=(243,173,141),font=font(13))
        flags=info["results"][i]["warnings"]
        line="MANUAL REVIEW REQUIRED" if not flags else f"QC NOTES: {len(flags)}"
        d.text((x,foot_y+23),line,fill=(177,202,226),font=font(12))
        status.append({"scene":scene["id"],"preview_mode":mode,"qc_warnings":flags})
    output.parent.mkdir(parents=True,exist_ok=True)
    out.save(output)
    manifest={"episode":episode.get("id"),"contact_sheet":str(output),
              "automated_likeness_verification":False,
              "visual_approval":False,"scenes":status}
    output.with_suffix(".json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--episode",type=Path,required=True)
    p.add_argument("--assets",type=Path,default=Path("assets"))
    p.add_argument("--output",type=Path,default=Path("dist/review_board.png"))
    args=p.parse_args()
    episode=json.loads(args.episode.read_text(encoding="utf-8"))
    result=contact_sheet(episode,args.assets,args.output)
    print(json.dumps(result,ensure_ascii=False))

if __name__=="__main__":
    main()
