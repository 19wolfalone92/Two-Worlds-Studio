"""Elizabeth Bones E01: private canonical-reference frames, six original WAV clocks.

Only source-authentic input scenes. No independent text-to-image sampling.
This module runs on CPU, including inside private Kaggle GPU workers.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont

SRC_SHA={
 "KOSTYA":"447373a1ee3162dd81ab29e07f5cbff24dbe4bab44ddbbc7394213ecb6ca0afd",
 "ELIZABETH":"7c691765a9f7cf3c190d110f114ce7d3b8a2f39eb681ecfe3056817c8e9a58a1",
}
VOICE_SHA={
 "S01":"58763966f8084cafd7a82221e7846c0f57b1ffe5234c4905afc4ce1691c5bc26",
 "S02":"d4be318dbe660ade3a0a0289a482244f5b336a66001f0bdfc1c16b5051964b33",
 "S03":"21c5bffae7bccdd179365cd0d139222d2ed484b74b7e7e792937b31d1d895296",
 "S04":"6fd367739292a710ade59887d10b2d7ec19a67d563eec19cbd2c1aff355f1965",
 "S05":"a047e6ecedbeb33524f41e37ac61e2d7785b9051c669318460edbbc6a37cd5e8",
 "S06":"a9e29b7fa792e840d8c9fc02775a119c52ea1527bc9d96624506b7663c69ce21"
}
VOICE_SECONDS={"S01":2.16,"S02":4.88,"S03":1.44,"S04":3.44,"S05":4.24,"S06":3.28}
DIALOGUES={
 "S01":"Следующий. Имя?",
 "S02":"Константин Мортисович Костин. Для друзей — Костя.",
 "S03":"Возраст?",
 "S04":"Двести три. Но душой я мальчишка!",
 "S05":"Вижу. А где душа?",
 "S06":"На испытательном. Уже три века.",
}
# I2V genuine 24 FPS; use only native video frames, never synthetic hold/loops.
# Wan2.2 length must be 1 modulo 4. All lengths include 0.2–0.4s padding.
SHOTS=(
 ("S00","KOSTYA",49,1.95,1.00,.48,.40,
  "Realistic gothic interview office, an anatomical ivory skeleton gentleman in a fitted matte black suit and black bow tie sits before a desk. His skull slowly tilts to the right and his bony hands shift delicately. Natural coherent motion in his neck, jaw, fingers and shoulders. Beautiful warm amber Victorian cinema lighting. NO flesh, no wardrobe changes, no flicker, no new characters."),
 ("S01","ELIZABETH",61,2.45,1.00,.47,.43,
  "Live action shot of the SAME brunette Elizabeth behind a walnut desk in a dark Victorian office, fitted black lace blouse. She looks down briefly at her papers, then raises an eyebrow, gently moves her shoulders and starts speaking with lifelike jaw, cheeks and eye motion. Consistent face and long brown hair. Cinematic warm tungsten lighting; NO face transformation, no outfit changes, no frozen body."),
 ("S02","KOSTYA",125,5.15,1.08,.51,.43,
  "Continuous five-second acting take: the SAME anatomically true ivory skeleton gentleman in a fitted matte black three-piece suit and small black bow tie. He answers a job interview confidently, his lower bony jaw articulates with speech, his skull turns slightly and bony fingers lift in one expressive but restrained gesture. The neck, shoulders and collar all move as connected living performance, not floating lips. No human skin, no suit color change, no new props, no RGB artifacts."),
 ("S03","ELIZABETH",41,1.70,1.18,.48,.37,
  "Cinematic close medium shot: EXACT same brunette actress with long dark hair and dark high-neck lace blouse. She looks up from a cream paper, arches her eyebrow skeptically and speaks a one-word question. Her face, eyes, cheeks, chin, neck and shoulders all have subtle natural continuous movement. No human identity change, no stylized animation, unchanged office and warm lighting."),
 ("S04","KOSTYA",89,3.70,1.12,.49,.43,
  "Three-second comedic answer, the same anatomically accurate ivory skull and bony hands dressed in the identical black wool suit and black bow tie. The skeleton shrugs theatrically with both shoulders and slightly lifts his right skeletal hand, opens his real bone jaw as though speaking, then settles back. Physically coherent anatomy and suit folds, richly lit walnut interview room. Never morph into a flesh-faced man, no detached jaw or neon stripes."),
 ("S05","ELIZABETH",109,4.53,1.08,.50,.47,
  "Four-second live-action close medium take of the identical young brunette interviewer in a matte black lace blouse at the same walnut desk with a brass lamp. She gives a long dry ironic look, shifts her weight and hand holding the pen, raises one eyebrow and asks an amused question; natural moving jaw, cheeks, eyes and shoulders. Consistent hairline, skin texture, wardrobe and warm burgundy background; realistic human actor performance."),
 ("S06","KOSTYA",89,3.70,1.08,.49,.42,
  "The same fully skeletal ivory-bone gentleman in an unchanged perfectly tailored BLACK suit and BLACK bow tie delivers a dry punchline. His bony lower jaw moves as he speaks, subtle skull tilt, shoulder roll and a small natural bony-hand gesture, then a perfectly still comedic beat. Keep original skull teeth and dark hollow eye sockets, no flesh, no new hat, no morphing, no temporal glitches; exquisite warm gothic cinema light."),
)
WIDTH,HEIGHT,FPS=320,576,24

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
 return h.hexdigest()

def gather_private(mount:Path, output:Path)->dict:
 """No secrets or unverified portraits may be used; output is PRIVATE only."""
 if output.exists(): raise FileExistsError("Do not overwrite private art folder")
 output.mkdir(parents=True)
 found={}
 for who,expected in SRC_SHA.items():
  files=list(mount.rglob(who+".png"))
  matches=[p for p in files if sha(p)==expected]
  if len(matches)!=1: raise ValueError("Canonical "+who+" source missing or ambiguous")
  with Image.open(matches[0]) as image:
   if image.format!="PNG" or image.size!=(704,1216):
    raise ValueError("Wrong original portrait dimension/format")
  found[who]=matches[0]
 for scene,expected in VOICE_SHA.items():
  files=list(mount.rglob(scene+".wav"))
  matches=[p for p in files if sha(p)==expected]
  if len(matches)!=1: raise ValueError("Missing exact original Qwen WAV "+scene)
  dst=output/"voices"
  dst.mkdir(exist_ok=True)
  shutil.copy2(matches[0],dst/(scene+".wav"))
 records={}
 for sid,who,frames,secs,zoom,cx,cy,prompt in SHOTS:
  with Image.open(found[who]) as original:
   img=original.convert("RGB")
   # Different actor framing from SAME original, never fresh T2I characters.
   iw,ih=img.size
   rw=int(iw/zoom);rh=int(ih/zoom)
   x=int((iw-rw)*cx*2);y=int((ih-rh)*cy*2)
   x=max(0,min(iw-rw,x));y=max(0,min(ih-rh,y))
   img=img.crop((x,y,x+rw,y+rh))
   frame=ImageOps.fit(img,(WIDTH,HEIGHT),method=Image.Resampling.LANCZOS)
   frame=ImageEnhance.Color(frame).enhance(.98)
   frame=ImageEnhance.Contrast(frame).enhance(1.03)
   frames_dir=output/"frames"
   frames_dir.mkdir(exist_ok=True)
   path=frames_dir/(sid+".png")
   frame.save(path,optimize=True)
   if path.stat().st_size<40000:raise ValueError("Tiny/blank source still "+sid)
  records[sid]={"source_character":who,"source_original_sha256":SRC_SHA[who],
   "reference_sha256":sha(path),"requested_frames":frames,
   "duration":secs,"motion_prompt":prompt,
   "caption":DIALOGUES.get(sid,""),"voice_sha256":VOICE_SHA.get(sid)}
 lock={"schema":7,"episode":"ELIZABETH_BONES_E01","model":"wan22-5b",
       "dimensions":[WIDTH,HEIGHT,FPS],"source_sha256":SRC_SHA,
       "scenes":records,"voice_sha256":VOICE_SHA,"generated_frames":False,
       "gpu_executed":False,"human_visual_review":False,"published":False}
 (output/"v7_source_lock.json").write_text(json.dumps(lock,ensure_ascii=False,indent=2)+"\n")
 return lock

def contact_sheet(root:Path,out:Path)->None:
 if out.exists():raise FileExistsError("Do not overwrite visual review")
 from PIL import ImageFont
 w,h=180,324
 canvas=Image.new("RGB",(w*4,(h+44)*2),(23,16,25))
 d=ImageDraw.Draw(canvas)
 for i,scene in enumerate(SHOTS):
  sid=scene[0];who=scene[1]
  with Image.open(root/"frames"/(sid+".png")) as image:
   image=image.convert("RGB").resize((w,h),Image.Resampling.LANCZOS)
   canvas.paste(image,((i%4)*w,(i//4)*(h+44)))
  d.text(((i%4)*w+8,(i//4)*(h+44)+h+7),
         sid+" — "+who,fill=(245,214,175))
 d.text((w+10,(h+44)+h+8),"UNAPPROVED — REVIEW ONLY",fill=(240,115,130))
 canvas.save(out,format="JPEG",quality=85,optimize=True)
