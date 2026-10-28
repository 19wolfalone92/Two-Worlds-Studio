"""Rights-cleared, scene-timed audio post-production for real episode masters.

No cloning of real voices, no unlicensed sound search, and no default paid APIs.
Real voices come from approved files supplied by the owner/Work.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
REGISTRY=ROOT/"audio/voice_registry.json"
ALLOWED={".wav",".flac",".mp3",".ogg",".m4a"}
MAX_AUDIO=60*1024*1024
RIGHTS={"original","owned","commissioned","licensed"}

def read_registry(path:Path=REGISTRY)->dict:
    data=json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema")!=1 or not isinstance(data.get("speakers"),dict):
        raise ValueError("Invalid voice registry")
    return data

def audio_asset(root:Path,relative:str)->Path:
    if not isinstance(relative,str) or not relative or Path(relative).is_absolute():
        raise ValueError("Audio path must be relative")
    if ".." in Path(relative).parts or Path(relative).suffix.lower() not in ALLOWED:
        raise ValueError("Unsupported or unsafe audio path")
    result=(root/relative).resolve()
    if root.resolve() not in result.parents or not result.is_file():
        raise ValueError("Audio file missing or outside assets/")
    if result.stat().st_size>MAX_AUDIO:
        raise ValueError("Audio exceeds size limit")
    return result

def describe_audio(path:Path)->dict:
    cmd=["ffprobe","-v","error","-select_streams","a:0",
         "-show_entries","stream=codec_name,channels,sample_rate,duration:format=duration",
         "-of","json",str(path)]
    data=json.loads(subprocess.run(cmd,capture_output=True,text=True,check=True).stdout)
    streams=data.get("streams",[])
    if not streams:
        raise ValueError("Audio file contains no sound stream")
    stream=streams[0]
    duration=float(stream.get("duration") or data.get("format",{}).get("duration",0))
    if not math.isfinite(duration) or not 0.05<=duration<=600:
        raise ValueError("Audio has invalid duration")
    return {"duration":duration,"channels":stream.get("channels"),
            "sample_rate":stream.get("sample_rate"),
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def _rights(meta:dict,where:str):
    if meta.get("review_status")!="approved":
        raise ValueError(f"{where}: audio requires explicit human approval")
    if meta.get("rights") not in RIGHTS or len(str(meta.get("rights_note","")).strip())<8:
        raise ValueError(f"{where}: audio source rights are undocumented")

def audit_audio(episode:dict,asset_root:Path,*,publish:bool=False,
                registry_path:Path=REGISTRY,fps:int=24)->dict:
    registry=read_registry(registry_path)
    offset=0
    tracks=[]
    for scene in episode.get("scenes",[]):
        scene_frames=max(1,round(float(scene["duration"])*fps))
        scene_duration=scene_frames/fps
        has_dialogue=bool(str(scene.get("voiceover","")).strip())
        voice=scene.get("voice")
        if publish and has_dialogue and not isinstance(voice,dict):
            raise ValueError(f"{scene['id']}: missing approved voice recording")
        if voice is not None:
            if not isinstance(voice,dict):
                raise ValueError(f"{scene['id']}: voice must be an object")
            speaker=voice.get("speaker")
            profile=registry["speakers"].get(speaker)
            if not isinstance(profile,dict):
                raise ValueError(f"{scene['id']}: unknown voice profile {speaker}")
            if voice.get("voice_version")!=profile.get("version"):
                raise ValueError(f"{scene['id']}: voice identity drift for {speaker}")
            person=scene.get("continuity",{}).get("occupant")
            allowed=[person]+scene.get("secondary_speakers",[])
            if person and speaker not in allowed:
                raise ValueError(f"{scene['id']}: voice speaker not in scene's acting consciousness")
            if publish and profile.get("review_status")!="approved":
                raise ValueError(f"{speaker}: voice casting has not been approved")
            if publish:
                _rights(voice,scene["id"])
            path=audio_asset(asset_root,voice.get("path"))
            info=describe_audio(path)
            if voice.get("sha256") is not None and voice["sha256"]!=info["sha256"]:
                raise ValueError(f"{scene['id']}: audio SHA-256 mismatch")
            offset_ms=int(voice.get("offset_ms",100))
            if not 0<=offset_ms<=1000:
                raise ValueError("Voice offset must be 0..1000 ms")
            if info["duration"]+offset_ms/1000>scene_duration+.08:
                raise ValueError(f"{scene['id']}: voiceover spills into the next scene")
            tracks.append({"kind":"voice","scene":scene["id"],"speaker":speaker,
                           "path":str(path),"start_ms":round(offset/fps*1000)+offset_ms,
                           "duration":info["duration"],"sha256":info["sha256"]})
        offset+=scene_frames
    music=episode.get("audio",{}).get("music")
    if music:
        if publish:
            _rights(music,"music")
        path=audio_asset(asset_root,music.get("path"))
        info=describe_audio(path)
        if music.get("sha256") is not None and music["sha256"]!=info["sha256"]:
            raise ValueError("Music hash mismatch")
        tracks.append({"kind":"music","path":str(path),"start_ms":0,
                       "duration":info["duration"],"sha256":info["sha256"]})
    if publish and not tracks:
        raise ValueError("Master audio requires approved music or dialogue")
    return {"total_seconds":offset/fps,"tracks":tracks,
            "audio_review_required":True,"approved":bool(publish)}

def mix_audio(video:Path,tracks:list[dict],*,output:Path=None)->Path:
    """Merge approved audio into an existing MP4. No network access."""
    if not tracks:
        return video
    out=output or video.with_name(video.stem+"_mix.mp4")
    if out.resolve()==video.resolve():
        raise ValueError("Do not mix in place")
    inputs=["-i",str(video)]
    graph=["[0:a]volume=0.12,aresample=48000,pan=stereo|c0=c0|c1=c0[bed]"]
    labels=["[bed]"]
    for index,track in enumerate(tracks,1):
        inputs+=["-i",track["path"]]
        delay=int(track["start_ms"])
        gain="1.0" if track["kind"]=="voice" else "0.16"
        graph.append(f"[{index}:a]aresample=48000,aformat=channel_layouts=stereo,"
                     f"adelay={delay}:all=1,volume={gain}[track{index}]")
        labels.append(f"[track{index}]")
    graph.append("".join(labels)+f"amix=inputs={len(labels)}:duration=first:normalize=0,"
                 "alimiter=limit=0.9[mix]")
    args=["ffmpeg","-nostdin","-y","-v","error",*inputs,
          "-filter_complex",";".join(graph),"-map","0:v:0","-map","[mix]",
          "-c:v","copy","-c:a","aac","-b:a","192k",
          "-ar","48000","-ac","2","-shortest","-movflags","+faststart",str(out)]
    subprocess.run(args,check=True)
    return out

def main():
    p=argparse.ArgumentParser(description="Audit approved dialogue and music supplied by Work")
    p.add_argument("--episode",type=Path,required=True)
    p.add_argument("--assets",type=Path,default=ROOT/"assets")
    p.add_argument("--publish-gate",action="store_true")
    args=p.parse_args()
    ep=json.loads(args.episode.read_text(encoding="utf-8"))
    print(json.dumps(audit_audio(ep,args.assets,publish=args.publish_gate),indent=2))

if __name__=="__main__":
    main()
