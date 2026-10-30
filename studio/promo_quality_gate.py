"""Offline promo QC for Two Worlds Studio / Elizabeth Bones.

Blocks false claims of a finished cinematic brand video when only still
images, soundtrack and captions have been assembled. No API spending,
automatic publishing, or access to private user photos.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

VIDEO={".mp4",".mov",".webm"}
AUDIO={".wav",".mp3",".flac",".m4a"}
PICTURE={".jpg",".jpeg",".png",".webp"}
WORLDS={"comic","care"}


class InvalidPromo(ValueError):
    pass


def bounded(base:Path,value:str,allowed:set)->Path:
    if not isinstance(value,str) or not value:
        raise InvalidPromo("Empty asset")
    rel=Path(value)
    if rel.is_absolute() or ".." in rel.parts or rel.suffix.lower() not in allowed:
        raise InvalidPromo(f"Unsafe asset reference: {value}")
    base=base.resolve()
    p=(base/rel).resolve()
    if base not in p.parents or not p.is_file():
        raise InvalidPromo(f"Missing/escaped asset: {value}")
    return p


def probe(path:Path)->dict:
    p=subprocess.run(["ffprobe","-v","error","-show_format","-show_streams",
                      "-of","json",str(path)],capture_output=True,text=True)
    if p.returncode:
        raise InvalidPromo("Unreadable media: "+path.name)
    return json.loads(p.stdout)


def media_duration(path:Path,kind:str)->float:
    data=probe(path)
    if not any(s.get("codec_type")==kind for s in data.get("streams",[])):
        raise InvalidPromo(f"No {kind} stream: {path.name}")
    seconds=float(data.get("format",{}).get("duration") or 0)
    if not math.isfinite(seconds) or seconds<.25:
        raise InvalidPromo(f"Invalid media duration: {path.name}")
    return seconds


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):
            h.update(c)
    return h.hexdigest()


def check_manifest(path:Path)->dict:
    """Film/voice provenance gate; technical PASS is NEVER publication approval."""
    path=Path(path).resolve()
    doc=json.loads(path.read_text(encoding="utf8"))
    if doc.get("schema")!=1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",str(doc.get("id",""))):
        raise InvalidPromo("Invalid project id/schema")
    if doc.get("language")!="ru" or doc.get("platform")!="instagram_reels":
        raise InvalidPromo("Require explicit Russian Instagram Reel")
    shots=doc.get("shots")
    if not isinstance(shots,list) or not 3<=len(shots)<=25:
        raise InvalidPromo("Require 3-25 actual shots")
    seen=set()
    length=motion=spoken=0
    failures=[]
    receipts=[]
    for i,shot in enumerate(shots,1):
        if not isinstance(shot,dict) or shot.get("world") not in WORLDS:
            raise InvalidPromo(f"Shot {i}: invalid world")
        seen.add(shot["world"])
        kind=shot.get("kind")
        if kind not in {"still","motion"}:
            raise InvalidPromo(f"Shot {i}: declare still or motion")
        p=bounded(path.parent,shot.get("asset"),VIDEO if kind=="motion" else PICTURE)
        seconds=shot.get("seconds")
        if type(seconds) not in (float,int) or not math.isfinite(seconds) or not .3<=seconds<=15:
            raise InvalidPromo(f"Shot {i}: bad duration")
        length+=seconds
        if kind=="motion":
            if media_duration(p,"video")+.05<seconds:
                raise InvalidPromo(f"Shot {i}: source too short")
            motion+=seconds
            if shot.get("motion_provenance") not in {"camera_footage","rendered_character_animation"}:
                failures.append(f"shot_{i}: a zoom/pan on a still is not real character motion")
        else:
            from PIL import Image
            with Image.open(p) as img:
                img.verify()
        if shot.get("rights_review")!="human_verified":
            failures.append(f"shot_{i}: unverified media rights")
        if shot.get("contains_real_deceased") is not False:
            failures.append(f"shot_{i}: cannot confirm exclusion of real deceased")
        if shot.get("dialogue_text"):
            if not shot.get("voice"):
                failures.append(f"shot_{i}: no real recorded/generated voice file")
            else:
                voice=bounded(path.parent,shot["voice"],AUDIO)
                media_duration(voice,"audio")
                spoken+=1
                if shot.get("voice_review")!="human_listened":
                    failures.append(f"shot_{i}: voice must be listened to")
        receipts.append({"shot":i,"source_sha256":sha(p),"kind":kind,"seconds":seconds})
    if seen!=WORLDS:
        failures.append("both comic and care worlds required")
    if not 12<=length<=60:
        failures.append("short promo must be 12-60 sec")
    if motion/max(length,.001)<.65:
        failures.append("real motion below 65%; still-photo slideshow cannot be cinematic master")
    if spoken<2:
        failures.append("real Russian spoken dialogue/narration missing")
    if not doc.get("final_card_text") or not doc.get("call_to_action"):
        failures.append("end card/CTA missing")
    if doc.get("music_review")!="human_listened":
        failures.append("audio rights and listening review pending")
    return {"schema":1,"id":doc["id"],
            "status":"BLOCKED" if failures else "TECHNICAL_PREFLIGHT_PASS",
            "ready_to_publish":False,"human_approval":"REQUIRED",
            "duration":round(length,2),"real_motion_ratio":round(motion/max(length,.001),3),
            "spoken_shots":spoken,"blocks":failures,"receipts":receipts,
            "note":"Human consent/voice quality/artistic quality cannot be proven by a manifest."}


def inspect_video(path:Path)->dict:
    """Conservative warning detector for music-only slide presentations.

    Optical differences alone cannot identify lip sync or genuine acting.
    Ken Burns makes frames change, so also check LOW AMPLITUDE motion.
    """
    import numpy as np
    data=probe(Path(path))
    v=next((s for s in data.get("streams",[]) if s.get("codec_type")=="video"),None)
    if not v:
        raise InvalidPromo("No video stream")
    w,h=int(v.get("width",0)),int(v.get("height",0))
    has_audio=any(s.get("codec_type")=="audio" for s in data.get("streams",[]))
    raw=subprocess.run(["ffmpeg","-nostdin","-v","error","-i",str(path),
        "-vf","fps=3,scale=180:320:flags=area,format=gray",
        "-t","60","-f","rawvideo","-pix_fmt","gray","pipe:1"],
        capture_output=True)
    if raw.returncode:
        raise InvalidPromo("Cannot decode video")
    unit=180*320
    frames=[np.frombuffer(raw.stdout[j:j+unit],dtype=np.uint8).reshape(320,180)
            for j in range(0,len(raw.stdout),unit) if len(raw.stdout[j:j+unit])==unit]
    if len(frames)<4:
        raise InvalidPromo("Insufficient sampled frames")
    low=stationary=almost_black=0
    for i,f in enumerate(frames):
        almost_black+=f.mean()<10
        if not i: continue
        p=frames[i-1][64:250,22:158].astype(np.int16)
        c=f[64:250,22:158].astype(np.int16)
        diff=np.abs(c-p)
        low+=diff.mean()<8
        stationary+=(diff>9).mean()<.018
    frac_low=low/(len(frames)-1)
    frac_still=stationary/(len(frames)-1)
    issues=[]
    if not has_audio: issues.append("No audio stream")
    if w*16!=h*9: issues.append("Not 9:16")
    if w<720 or h<1280: issues.append("Below review resolution")
    if frac_low>.72:
        issues.append(f"{frac_low:.0%} low-amplitude intervals: likely slides/Ken Burns")
    if frac_still>.5:
        issues.append(f"{frac_still:.0%} stationary intervals")
    if almost_black/len(frames)>.12:
        issues.append("Excessive black frames")
    return {"status":"WARNINGS" if issues else "NO_BASIC_DEFECTS_DETECTED",
            "issues":issues,"low_visual_change_ratio":round(frac_low,3),
            "stationary_ratio":round(frac_still,3),
            "resolution":[w,h],"audio_stream":has_audio,
            "ready_to_publish":False,
            "note":"Warning heuristic cannot certify acting, voice, identity or rights."}


def main(argv=None):
    p=argparse.ArgumentParser(description="Offline Elizabeth Bones cinema gate")
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest",type=Path)
    g.add_argument("--video",type=Path)
    p.add_argument("--out",type=Path)
    a=p.parse_args(argv)
    result=check_manifest(a.manifest) if a.manifest else inspect_video(a.video)
    txt=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(txt,encoding="utf8")
    print(txt)


if __name__=="__main__":
    main()
