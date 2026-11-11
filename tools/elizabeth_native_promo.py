"""Native TWO WORLDS STUDIO Elizabeth Bones EB01 18s review build.

The only editor is studio.hybrid_master. InVideo is NOT used.
Production inputs are private media in --media, never committed to Git.
--fixture is a *technical storyboard only* with unapproved synthetic artwork,
no voices and no neural motion. Do not mistake its QC for a finished film.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SHOTS = (
    ("S00", 2.20, "", "both"),
    ("S01", 2.125, "Следующий. Имя?", "elizabeth"),
    ("S02", 4.250, "Константин Мортисович Костин. Для друзей — Костя.", "kostya"),
    ("S03", 1.600, "Возраст?", "elizabeth"),
    ("S04", 3.050, "Двести три. Но душой я мальчишка!", "kostya"),
    ("S05", 2.350, "Вижу. А где душа?", "elizabeth"),
    ("S06", 2.900, "На испытательном. Уже три века.", "kostya"),
)
ART_EXT = (".png", ".jpg", ".jpeg", ".webp")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a")
W, H = 576, 1024


def media_file(root: Path, name: str, extensions: tuple[str, ...], required: bool = False) -> Path | None:
    matches = [root / (name + ext) for ext in extensions if (root / (name + ext)).is_file()]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous duplicate sources: {name}")
    if not matches:
        if required:
            raise FileNotFoundError("Missing original artwork, never synthesize it silently: " + name)
        return None
    p = matches[0]
    if p.is_symlink() or p.stat().st_size < 100:
        raise ValueError("Unsafe/small media file: " + name)
    return p


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=30)
    return float(json.loads(out.stdout)["format"]["duration"])


def draw_fixture(target: Path) -> None:
    """Create *clearly labeled* dummy frames for a real end-to-end studio CI build."""
    target.mkdir(parents=True, exist_ok=True)
    font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    def person(d: ImageDraw.ImageDraw, kind: str, x: int, y: int) -> None:
        # Consistent symbolic characters, intentionally NOT a photoreal AI performance.
        if kind == "kostya":
            d.polygon([(x-135,y+135),(x-32,y+92),(x+47,y+92),(x+132,y+132),
                       (x+150,H*.80),(x-152,H*.80)],fill=(19,21,27))
            d.polygon([(x-35,y+110),(x,y+205),(x+32,y+110)],fill=(208,197,168))
            d.polygon([(x-8,y+170),(x+9,y+170),(x+28,y+290),(x-1,y+310)],fill=(116,36,61))
            d.ellipse((x-76,y-107,x+77,y+67),fill=(218,205,177),outline=(109,88,65),width=4)
            d.ellipse((x-58,y-42,x-13,y+1),fill=(27,24,28))
            d.ellipse((x+14,y-42,x+58,y+1),fill=(27,24,28))
            d.polygon([(x-7,y+7),(x+8,y+7),(x+17,y+30),(x-11,y+30)],fill=(53,44,44))
            d.rounded_rectangle((x-51,y+40,x+51,y+81),radius=9,fill=(212,197,167))
            for j in range(-3,4):
                d.line((x+j*13,y+44,x+j*13,y+76),fill=(96,84,74),width=3)
            d.ellipse((x-100,y-124,x+100,y-91),fill=(13,15,18))
            d.rounded_rectangle((x-64,y-167,x+64,y-112),radius=10,fill=(13,15,18))
            d.rectangle((x-64,y-129,x+64,y-116),fill=(117,38,63))
        else:
            d.polygon([(x-110,y+145),(x-35,y+92),(x+52,y+92),(x+141,y+156),
                       (x+158,H*.80),(x-156,H*.80)],fill=(24,22,30))
            for a in range(-120,130,20):
                d.line((x+a,y+225,x+a*1.15,H*.80),fill=(52,49,57),width=2)
            d.ellipse((x-94,y-134,x+95,y+193),fill=(36,23,31))
            d.ellipse((x-52,y-109,x+61,y+101),fill=(211,157,136),outline=(119,77,69),width=4)
            d.polygon([(x-74,y-116),(x+51,y-126),(x+59,y-65),(x+2,y-88),(x-55,y-33)],fill=(32,22,28))
            d.arc((x-31,y-29,x-2,y-1),178,349,fill=(53,41,43),width=4)
            d.arc((x+14,y-28,x+39,y-1),178,349,fill=(53,41,43),width=4)
            d.arc((x-19,y+43,x+28,y+59),12,169,fill=(99,48,52),width=4)
            d.ellipse((x-11,y+150,x+14,y+179),fill=(151,29,60),outline=(191,156,98),width=3)
    for index,(sid,_,_,focus) in enumerate(SHOTS):
        import random
        random.seed(44)
        im=Image.new("RGB",(W,H))
        px=im.load()
        for y in range(H):
            for x in range(W):
                v=1.0-.63*((x-W*.5)/(W*.8))**2
                grain=random.randrange(-3,4)
                px[x,y]=(max(0,int((38+12*y/H)*v)+grain),
                         max(0,int((28+7*y/H)*v)+grain),
                         max(0,int((37+7*y/H)*v)+grain))
        d=ImageDraw.Draw(im)
        d.rounded_rectangle((74,94,W-74,H-243),radius=100,fill=(30,37,45),outline=(132,108,82),width=8)
        for xx in range(135,W-85,83):
            d.line((xx,165,xx,H-300),fill=(70,69,77),width=4)
        d.polygon([(0,H*.76),(W,H*.72),(W,H),(0,H)],fill=(34,22,27))
        d.line((0,H*.77,W,H*.735),fill=(129,96,71),width=12)
        d.line((80,H*.47,142,H*.32),fill=(172,130,76),width=8)
        d.polygon([(99,H*.30),(180,H*.30),(198,H*.37),(78,H*.37)],fill=(172,121,69))
        d.ellipse((W*.79,H*.69,W*.85,H*.75),fill=(147,27,56))
        d.line((W*.81,H*.73,W*.83,H*.82),fill=(53,94,62),width=4)
        if focus == "both":
            person(d,"elizabeth",round(W*.31),round(H*.43))
            person(d,"kostya",round(W*.77),round(H*.42))
        else:
            person(d,focus,round(W*(.44 if index%2 else .57)),round(H*.43))
        # Visible QA watermark: the footage can never be mistaken for approved art.
        d.rectangle((0,0,W,110),fill=(20,15,22))
        d.text((W//2,27),"ELIZABETH BONES",font=ImageFont.truetype(bold,32),
               anchor="mt",fill=(239,208,175))
        d.text((W//2,74),"СОБЕСЕДОВАНИЕ · ТЕСТ СТУДИИ",
               font=ImageFont.truetype(font,17),anchor="mt",fill=(240,215,194))
        d.text((W-15,H-20),"DUMMY STORYBOARD · NOT FINAL",
               font=ImageFont.truetype(font,12),anchor="ra",fill=(208,145,111))
        im.save(target / (sid+".png"))


def manifest_for(media: Path, require_all_voices: bool) -> tuple[dict,list[str]]:
    if not media.is_dir() or media.is_symlink():
        raise ValueError("Expected private source directory")
    missing=[]
    shots=[]
    for sid,length,caption,focus in SHOTS:
        artwork=media_file(media,sid,ART_EXT,required=True)
        with Image.open(artwork) as im:
            if im.width < 400 or im.height < 700 or abs(im.width / im.height - 9 / 16) > .035:
                raise ValueError("Shot source fails vertical art size/aspect validation: "+sid)
        shot={"image":artwork.name,"duration":length,"caption":caption,
              "camera":"pan_left" if sid in {"S02","S06"} else "push_in",
              "camera_strength":.20,"shot_type":"wide" if sid=="S00" else "medium"}
        if caption:
            voice=media_file(media,"voice_"+sid,AUDIO_EXT)
            if voice is None:
                missing.append(sid)
            else:
                if duration(voice) > length + .14:
                    raise ValueError(f"Speech {sid} is longer than its locked shot")
                shot["voice"]=voice.name
        shots.append(shot)
    if require_all_voices and missing:
        raise ValueError("Not a full-voice preview; missing: "+",".join(missing))
    return {"schema":1,"delivery_profile":"mobile","language":"ru",
            "burn_subtitles":True,"editorial_approved":False,
            "rights_approved":False,"shots":shots},missing


def build(media:Path,output:Path,*,fixture:bool=False,require_all_voices:bool=False):
    media=media.resolve()
    if fixture:
        draw_fixture(media)
    manifest,missing=manifest_for(media,require_all_voices)
    source=media/"EB01_NATIVE_REVIEW_manifest.json"
    source.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    from studio.hybrid_master import produce
    result=produce(source,output.resolve(),master=False,quality_check=False)
    result.update(native_studio_build=True,fixture_art=fixture,
                  actual_neural_actor_motion=False,invideo_used=False,
                  missing_spoken_dialogue_shots=missing,production_approved=False)
    output.with_suffix(".studio_provenance.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--fixture",action="store_true")
    parser.add_argument("--require-all-voices",action="store_true")
    args=parser.parse_args()
    print(json.dumps(build(args.media,args.output,fixture=args.fixture,
                           require_all_voices=args.require_all_voices),
                     ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
