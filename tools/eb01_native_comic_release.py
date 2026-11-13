"""Explicit *motion-comic* cut for Elizabeth Bones E01, native TWO WORLDS STUDIO.

Not falsely labeled as seven neural acting takes. The original two owner
portraits and six Qwen3-RU recordings are cryptographically verified and stay
private. Authored graphic layouts + smooth bicubic camera, without fake lips,
face deform, InVideo, GPU, subscription or automated publication.
"""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import soundfile as sf

from studio.eb01_v7_storyboard import SRC_SHA, VOICE_SHA, VOICE_SECONDS, DIALOGUES, gather_private, sha
from studio.hybrid_master import produce

W,H,FPS=576,1024,24
FONT_PATHS=[
 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
BOLD_PATHS=[
 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
 "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"]

def font(size:int,bold=False):
    for name in (BOLD_PATHS if bold else FONT_PATHS):
        if Path(name).is_file():return ImageFont.truetype(name,size)
    raise RuntimeError("Missing Cyrillic DejaVu font: no garbled Russian titles")

def scene_art(base:Image.Image,output:Path,tag:str,variant:int,title:str="",body:str=""):
    """A real designed print-like panel from locked owner character photos."""
    canvas=Image.new("RGB",(W,H),(16,12,18))
    img=base.convert("RGB")
    cover=ImageOps.fit(img,(W-36,850),Image.Resampling.LANCZOS,
                       centering=(.51,.38 if variant==2 else .48))
    if variant==2:
        # Distinct close camera portrait: head/eyes/shoulders rather than
        # reusing the exact same distant shot. Still source authentic.
        close=img.crop((int(img.width*.12), 0,
                        int(img.width*.88), int(img.height*.58)))
        cover=ImageOps.fit(close,(W-36,850),Image.Resampling.LANCZOS,
                           centering=(.5,.15))
    canvas.paste(cover,(18,80))
    d=ImageDraw.Draw(canvas)
    # Simple gothic editorial borders, typography, and an intentional frame.
    for i,c in enumerate(("#af805e","#6c4150")):
        pad=8+i*5
        d.rectangle((pad,55+pad,W-pad,947-pad),outline=c,width=2)
    d.rectangle((0,0,W,76),fill=(22,15,25))
    d.text((27,22),"ELIZABETH BONES",font=font(30,True),fill=(221,184,145))
    d.rectangle((0,932,W,H),fill=(24,16,27))
    d.text((25,953),tag,font=font(21,True),fill=(222,176,125))
    if title: d.text((25,981),title,font=font(17),fill=(228,213,205))
    if body:
        d.rounded_rectangle((32,695,W-32,854),radius=10,fill=(26,18,29),outline=(161,117,92),width=2)
        for idx,line in enumerate(body.split("|")):
            d.text((50,716+idx*45),line,font=font(24 if idx==0 else 21,idx==0),fill=(246,217,194))
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output,optimize=True)

def card(out:Path,heading:str,lines:tuple[str,...],image:Image.Image|None=None):
    canvas=Image.new("RGB",(W,H),(19,14,26))
    d=ImageDraw.Draw(canvas)
    for i in range(5):
        d.rectangle((15+i*7,15+i*9,W-15-i*7,H-15-i*9),outline=(75+i*18,53+i*14,61+i*10),width=1)
    if image is not None:
        # Preserve the forehead/skull: PIL's default center crop cut off
        # the entire upper face in intro and both dossier title cards.
        crop=ImageOps.fit(image.convert("RGB"),(W-110,450),
                          Image.Resampling.LANCZOS,centering=(.5,.03))
        canvas.paste(crop,(55,92))
    d=ImageDraw.Draw(canvas)
    d.text((45,60 if image is None else 575),heading,font=font(31,True),fill=(239,201,161))
    y=175 if image is None else 665
    for line in lines:
        d.text((45,y),line,font=font(24),fill=(223,203,192))
        y+=58
    d.line((45,H-102,W-45,H-102),fill=(182,133,93),width=2)
    d.text((45,H-83),"ДВА МИРА  •  ОДНА ИСТОРИЯ",font=font(17),fill=(196,157,129))
    out.parent.mkdir(exist_ok=True,parents=True)
    canvas.save(out,optimize=True)

def quiet_score(out:Path,seconds:float):
    """Very low-level original noir pad; voice always takes precedence."""
    rate=24000
    n=int(math.ceil(seconds*rate))
    t=np.arange(n,dtype=np.float64)/rate
    music=np.zeros(n,dtype=np.float64)
    # Detuned low strings / soft organ, fixed deterministic arrangement.
    for frequency in (55.,65.406,82.407,110.):
        for harmonic,attenuation in ((1,.014),(2,.003),(3,.001)):
            phase=2*math.pi*frequency*harmonic*t
            music+=attenuation*np.sin(phase)*(0.78+0.22*np.sin(2*math.pi*.085*t))
    # Quiet tick accents on off-beats, no sudden transients.
    for when in np.arange(1.2,seconds,1.6):
        start=int(when*rate)
        end=min(n,start+int(.08*rate))
        u=np.arange(end-start)/rate
        music[start:end]+=.005*np.sin(2*math.pi*850*u)*np.exp(-52*u)
    fade=np.minimum(1,np.minimum(t/1.5,(seconds-t)/1.8))
    music=np.clip(music*np.maximum(0,fade),-.35,.35)
    sf.write(out,music.astype(np.float32),rate)

def produce_comic(dataset:Path,work:Path,output:Path)->dict:
    if work.exists() or output.exists():raise FileExistsError("Never overwrite original or cut")
    work.mkdir(parents=True)
    source=work/"originals"
    lock=gather_private(dataset,source)
    if lock["source_sha256"]!=SRC_SHA or lock["voice_sha256"]!=VOICE_SHA:
        raise RuntimeError("Original private references/voices altered")
    with Image.open(source/"frames"/"S01.png") as el, Image.open(source/"frames"/"S02.png") as ko:
        el,ko=el.convert("RGB"),ko.convert("RGB")
    art=work/"art"
    card(art/"intro.png","СОБЕСЕДОВАНИЕ",("Кто приходит работать","в ритуальное бюро?"),el)
    card(art/"outro.png","ПРОДОЛЖЕНИЕ СЛЕДУЕТ",("Элизабет и Костя","их знакомство только началось…"),ko)
    card(art/"dossier.png","ЛИЧНОЕ ДЕЛО № 203",("Имя: Константин","Фамилия: Костин","Стаж: два века"),ko)
    card(art/"resume.png","КАНДИДАТ",("Возраст: 203","Душа: на испытательном","Контракт: бессрочный"),el)
    for sid in ("S00","S01","S02","S03","S04","S05","S06"):
        who=lock["scenes"][sid]["source_character"]
        base=el if who=="ELIZABETH" else ko
        scene_art(base,art/(sid+".png"),
                  "ЭЛИЗАБЕТ  /  СОБЕСЕДОВАНИЕ" if who=="ELIZABETH" else "КОСТЯ  /  КАНДИДАТ",
                  1, title="Знакомство в ритуальном бюро")
        scene_art(base,art/(sid+"_close.png"),
                  "ЭЛИЗАБЕТ  /  ВОПРОС" if who=="ELIZABETH" else "КОСТЯ  /  ОТВЕТ",
                  2, title="Сцена из графической новеллы")
    shots=[]
    def shot(path,secs,camera="push_in",strength=.33,**kw):
        row={"image":"art/"+path,"duration":round(secs,3),"shot_type":"medium",
             "camera":camera,"camera_strength":strength,
             "caption_position":"bottom","approved":False,**kw}
        shots.append(row)
    shot("intro.png",3.6,"pull_out",.22)
    shot("S00.png",2.1,"pan_left",.21)
    shot("S01.png",VOICE_SECONDS["S01"]+.8,"push_in",.20,
         voice="originals/voices/S01.wav",voice_offset=.28,
         dialogue_text=DIALOGUES["S01"],caption=DIALOGUES["S01"])
    shot("dossier.png",1.65,"pan_right",.19)
    shot("S02.png",VOICE_SECONDS["S02"]+.75,"pull_out",.27,
         voice="originals/voices/S02.wav",voice_offset=.23,
         dialogue_text=DIALOGUES["S02"],caption=DIALOGUES["S02"])
    shot("S02_close.png",1.25,"push_in",.22)
    shot("S03.png",VOICE_SECONDS["S03"]+.85,"pan_left",.20,
         voice="originals/voices/S03.wav",voice_offset=.25,
         dialogue_text=DIALOGUES["S03"],caption=DIALOGUES["S03"])
    shot("resume.png",1.45,"push_in",.15)
    shot("S04.png",VOICE_SECONDS["S04"]+.85,"pan_right",.27,
         voice="originals/voices/S04.wav",voice_offset=.20,
         dialogue_text=DIALOGUES["S04"],caption=DIALOGUES["S04"])
    shot("S04_close.png",1.0,"pull_out",.13)
    shot("S05.png",VOICE_SECONDS["S05"]+.85,"push_in",.20,
         voice="originals/voices/S05.wav",voice_offset=.24,
         dialogue_text=DIALOGUES["S05"],caption=DIALOGUES["S05"])
    shot("S05_close.png",1.0,"pan_left",.15)
    shot("S06.png",VOICE_SECONDS["S06"]+1.2,"pull_out",.25,
         voice="originals/voices/S06.wav",voice_offset=.22,
         dialogue_text=DIALOGUES["S06"],caption=DIALOGUES["S06"])
    shot("outro.png",3.7,"push_in",.19)
    duration=sum(item["duration"] for item in shots)
    quiet_score(work/"score.wav",duration+.5)
    manifest={"schema":1,"width":W,"height":H,"fps":FPS,
              "language":"ru","visual_profile":"stylized_cinematic",
              "burn_subtitles":True,"music":"score.wav","music_volume":.09,
              "shots":shots,"editorial_approved":False,"rights_approved":False}
    mf=work/"motion_comic_manifest.json"
    mf.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    output.parent.mkdir(parents=True,exist_ok=True)
    result=produce(mf,output,master=False,quality_check=False)
    if result["static_camera_shots"]!=len(shots) or result["motion_clips"]!=0 or result["shots"]!=len(shots):
        raise RuntimeError("This is a motion comic, do not misstate neural motion")
    if result["audio_tracks"]<7 or not result["captions_burned_in"]:
        raise RuntimeError("Every Russian dialogue and score must be included")
    check=subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(output),"-f","null","-"],
                         capture_output=True,timeout=180)
    if check.returncode:raise RuntimeError("Final motion comic cannot fully decode")
    srt=output.with_suffix(".srt").read_text(encoding="utf-8")
    if srt.count("-->")!=6 or any(line not in srt for line in DIALOGUES.values()):
        raise RuntimeError("Six original spoken subtitles missing")
    receipt={"schema":1,"status":"COMPLETE_NATIVE_MOTION_COMIC_OWNER_REVIEW",
             "format":"AUTHORED_MOTION_COMIC_NOT_NEURAL_ACTING",
             "native_studio":"studio.hybrid_master.produce","original_private_actors":2,
             "original_russian_Qwen3_voice_lines":6,"subtitles":6,
             "original_video_shots_claimed":0,"designed_comic_shots":len(shots),
             "video_sha256":sha(output),"source_original_sha":SRC_SHA,
             "runtime_seconds":result["duration_seconds"],"published":False,
             "artistic_approved":False,"may_not_be_relabelled_neural_v7":True}
    output.with_suffix(".cut_qa.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    return receipt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(produce_comic(a.dataset,a.workspace,a.output),ensure_ascii=False,indent=2))

if __name__=="__main__":main()
