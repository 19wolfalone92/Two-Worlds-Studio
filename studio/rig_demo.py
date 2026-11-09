"""Render an honest *technical* demo of articulated character motion.

All character art is synthetic procedural test artwork. This is not
the actual Mira/Lyra, a finished TV shot, or an automatic art rigging model.
The proof is the motion hierarchy inside the real TWO WORLDS compositor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from PIL import Image,ImageDraw,ImageFilter
from studio.cinematic import CinematicPanel

W,H=360,640

def layer(drawer,name,fn,root):
    image=Image.new("RGBA",(W,H),(0,0,0,0))
    d=ImageDraw.Draw(image)
    fn(d)
    path=root/(name+".png")
    image.save(path)
    return {"name":name,"path":path.name}

def scenery(root):
    bg=Image.new("RGB",(W,H),(16,20,37))
    px=bg.load()
    for y in range(H):
        for x in range(W):
            glow=max(0,1-math.hypot(x-276,y-266)/290)
            px[x,y]=(int(16+24*glow+y/H*7),int(21+27*glow),
                     int(38+50*glow))
    d=ImageDraw.Draw(bg)
    d.polygon([(0,415),(W,445),(W,H),(0,H)],fill=(32,38,53))
    for y in (460,500,550,605):
        d.line([(0,y),(W,y+19)],fill=(54,61,74),width=2)
    for x in range(-30,W,55):
        d.line([(x,437),(x-52,H)],fill=(51,57,74),width=1)
    d.ellipse((226,130,338,430),fill=(66,73,111),outline=(145,163,207),width=7)
    d.ellipse((238,146,324,413),fill=(28,56,94),outline=(88,121,175),width=3)
    d.arc((245,155,322,405),80,280,fill=(153,177,232),width=3)
    bg.save(root/"background.png")
    return bg

def _art(root):
    body=layer(None,"torso",lambda d:(
        d.polygon([(112,290),(156,281),(177,310),(172,438),(99,439),(102,325)],fill=(75,84,109,255)),
        d.polygon([(113,298),(158,292),(161,348),(109,348)],fill=(102,107,132,255)),
        d.line([(110,345),(167,347)],fill=(190,143,130,255),width=5)),root)
    cloak=layer(None,"cloak",lambda d:(
        d.polygon([(110,291),(157,290),(176,480),(94,487),(89,410)],fill=(88,47,77,235)),
        d.line([(110,294),(95,471)],fill=(154,94,106,230),width=4)),root)
    left_leg=layer(None,"left_leg",lambda d:(
        d.polygon([(111,425),(139,427),(136,541),(111,544)],fill=(45,52,70,255)),
        d.polygon([(110,535),(139,540),(137,562),(99,564)],fill=(40,41,57,255))),root)
    right_leg=layer(None,"right_leg",lambda d:(
        d.polygon([(143,426),(164,425),(174,541),(151,545)],fill=(47,53,71,255)),
        d.polygon([(153,536),(175,535),(190,556),(149,564)],fill=(34,38,49,255))),root)
    head=layer(None,"head",lambda d:(
        d.ellipse((112,230,161,299),fill=(185,141,124,255),outline=(68,53,58,255),width=2),
        d.polygon([(110,240),(116,220),(155,218),(166,245),(159,258),(153,236),(122,238)],
                  fill=(48,39,55,255)),
        d.line([(143,264),(151,264)],fill=(42,43,50,255),width=2),
        d.line([(120,264),(129,263)],fill=(42,43,50,255),width=2),
        d.arc((127,270,143,285),5,80,fill=(113,84,82,255),width=2)),root)
    upper=layer(None,"upper_arm",lambda d:(
        d.polygon([(154,298),(175,304),(201,381),(181,394),(167,333)],
                  fill=(79,87,108,255),outline=(125,126,142,255)),
        d.ellipse((182,376,200,401),fill=(106,110,132,255))),root)
    forearm=layer(None,"forearm",lambda d:(
        d.polygon([(187,382),(204,383),(229,449),(213,457)],
                  fill=(94,99,120,255),outline=(149,137,146,255)),
        d.ellipse((208,446,239,468),fill=(197,149,128,255))),root)
    return [cloak,left_leg,right_leg,body,head,upper,forearm]

def make_scene(root):
    scenery(root)
    paths={p["name"]:p["path"] for p in _art(root)}
    rig={
      "schema":1,"root_pivot":[.38,.80],
      "parts":[
        {"name":"cloak","path":paths["cloak"],"pivot":[.31,.46],
         "z":0,"parent":"torso"},
        {"name":"left_leg","path":paths["left_leg"],"pivot":[.35,.67],"z":1},
        {"name":"right_leg","path":paths["right_leg"],"pivot":[.44,.67],"z":1},
        {"name":"torso","path":paths["torso"],"pivot":[.38,.53],"z":2},
        {"name":"head","path":paths["head"],"pivot":[.38,.45],"parent":"torso","z":4},
        {"name":"upper_arm","path":paths["upper_arm"],"pivot":[.46,.49],
         "parent":"torso","z":5},
        {"name":"forearm","path":paths["forearm"],"pivot":[.54,.61],
         "parent":"upper_arm","z":6}
      ],
      "tracks":{
        "__root__":[{"at":0,"x":-.12},{"at":.52,"x":.08},{"at":1,"x":.08}],
        "torso":[{"at":0,"angle":-2},{"at":.4,"angle":2},{"at":1,"angle":4}],
        "head":[{"at":0,"angle":-8},{"at":.5,"angle":6},{"at":1,"angle":-4}],
        "upper_arm":[{"at":0,"angle":8},{"at":.50,"angle":8},{"at":1,"angle":-54}],
        "forearm":[{"at":0,"angle":0},{"at":.52,"angle":0},{"at":1,"angle":-23}],
        "left_leg":[{"at":0,"angle":-10},{"at":.25,"angle":7},{"at":.5,"angle":0},{"at":1,"angle":0}],
        "right_leg":[{"at":0,"angle":8},{"at":.25,"angle":-9},{"at":.5,"angle":0},{"at":1,"angle":0}],
        "cloak":[{"at":0,"angle":-4},{"at":.3,"angle":4},{"at":1,"angle":-3}]
      },
      "environment_light":{"rgb":[88,112,169],"strength":.07},
      "ground_shadow":{"center":[.38,.875],"radii":[.13,.019],
                        "opacity":.38,"blur":.012}
    }
    Image.new("RGBA",(W,H),(0,0,0,0)).save(root/"actor_rest.png")
    return {"id":"TECH_DEMO","duration":8,
     "cinematic":{"layers":[
        {"role":"background","path":"background.png","depth":-.9},
        {"role":"character","path":"actor_rest.png","depth":.3,"rig":rig}],
       "camera":{"start":{"x":-.02,"y":0,"zoom":1.0},
                 "end":{"x":.04,"y":-.012,"zoom":1.07}},
       "vignette":.15,"grain":0}}

def render_tech_demo(output:Path,*,fps=12,seconds=8)->dict:
    if fps<4 or fps>30 or seconds<2 or seconds>12:
        raise ValueError("Demo bounded to 2..12 seconds, 4..30 FPS")
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp)
        scene=make_scene(folder)
        panel=CinematicPanel(scene,folder,W,H)
        output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists():
            raise FileExistsError("Will not overwrite technical demo")
        cmd=["ffmpeg","-nostdin","-y","-v","error","-f","rawvideo","-pix_fmt","rgb24",
             "-s",f"{W}x{H}","-r",str(fps),"-i","-","-an",
             "-c:v","libx264","-pix_fmt","yuv420p","-preset","veryfast","-crf","21",
             "-movflags","+faststart",str(output)]
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        frames=round(fps*seconds)
        try:
            for n in range(frames):
                image=panel.frame(n/max(1,frames-1))
                process.stdin.write(image.tobytes())
            process.stdin.close()
            err=process.stderr.read()
            exitcode=process.wait(timeout=120)
            if exitcode:
                raise RuntimeError(err.decode(errors="replace")[-2000:])
        except BaseException:
            process.kill()
            output.unlink(missing_ok=True)
            raise
        manifest={"art":"procedural ONLY, not a TWO WORLDS episode",
          "purpose":"proves articulated sprite keyframes and live cinematic compositing",
          "width":W,"height":H,"fps":fps,"frames":frames,
          "characters":"synthetic / noncanonical","engine":"studio.character_engine",
          "human_approved":False,
          "sha256":hashlib.sha256(output.read_bytes()).hexdigest()}
        output.with_suffix(".json").write_text(json.dumps(manifest,indent=2)+"\n")
        return manifest

def main():
    p=argparse.ArgumentParser(description="Technical rig demo (NOT professional episode art)")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fps",type=int,default=12)
    p.add_argument("--seconds",type=int,default=8)
    args=p.parse_args()
    print(json.dumps(render_tech_demo(args.output,fps=args.fps,seconds=args.seconds),indent=2))
if __name__=="__main__":
    main()
