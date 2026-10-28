"""Kokoro 82M: offline-capable voice draft rendering, not an automatic approval.

Uses only explicitly supplied text or text spoken in a Work-produced scene's
tts field. No story generation, voice cloning or public publication.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
CAST=ROOT/"audio/kokoro_cast.json"
VOICE_REGISTRY=ROOT/"audio/voice_registry.json"
SPEAKER_ID=re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,32}$")
SCENE_ID=re.compile(r"^S\d{2,4}$")

def load_cast(path:Path=CAST)->dict:
    data=json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema")!=1 or not isinstance(data.get("speakers"),dict):
        raise ValueError("Invalid Kokoro voice cast registry")
    for name,profile in data["speakers"].items():
        if not SPEAKER_ID.fullmatch(name):
            raise ValueError("Invalid speaker ID")
        if not isinstance(profile,dict) or profile.get("language") not in ("a","b"):
            raise ValueError(f"{name}: unknown Kokoro language")
        if not re.fullmatch(r"[ab][fm]_[a-z0-9_]{2,40}",profile.get("voice","")):
            raise ValueError(f"{name}: invalid Kokoro voice profile")
    return data

def jobs_from_episode(episode:dict,cast:dict)->list[dict]:
    scenes=episode.get("scenes")
    if not isinstance(scenes,list) or not scenes:
        raise ValueError("Work episode has no scenes")
    jobs=[]
    for scene in scenes:
        d=scene.get("tts")
        if d is None:
            continue
        if not isinstance(d,dict):
            raise ValueError("tts must be explicit object with speaker and text")
        speaker=d.get("speaker")
        text=d.get("text")
        if speaker not in cast["speakers"]:
            raise ValueError(f"{scene.get('id')}: speaker has no stable Kokoro casting")
        if not isinstance(text,str) or not 1<=len(text.strip())<=550:
            raise ValueError("Text length outside single-scene voice limit")
        sid=scene.get("id")
        if not isinstance(sid,str) or not SCENE_ID.fullmatch(sid):
            raise ValueError("Unrecognized Work scene ID")
        # Consistency: speaking consciousness, never the physical body.
        occupant=scene.get("continuity",{}).get("occupant")
        other=scene.get("secondary_speakers",[])
        if occupant and speaker not in ([occupant]+other):
            raise ValueError(f"{sid}: speaker not represented in this Work scene")
        jobs.append({"scene":sid,"speaker":speaker,"text":text.strip(),
                     "voice":cast["speakers"][speaker]["voice"],
                     "language":cast["speakers"][speaker]["language"]})
    return jobs

def synthesize(text:str,voice:str,language:str,output:Path)->dict:
    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline
    except ImportError as exc:
        raise RuntimeError("Kokoro is missing; install 'kokoro>=0.9.2 soundfile' and espeak-ng") from exc
    pipeline=KPipeline(lang_code=language)
    chunks=[]
    # KPipeline may emit multiple segments. Concatenate with short neutral breaks.
    for _graphemes,_phones,audio in pipeline(text,voice=voice):
        block=np.asarray(audio,dtype=np.float32).reshape(-1)
        if not len(block):
            continue
        if chunks:
            chunks.append(np.zeros(6000,dtype=np.float32)) # 250ms @ 24k
        chunks.append(block)
    if not chunks:
        raise RuntimeError("Kokoro produced no audible waveform")
    pcm=np.concatenate(chunks)
    if len(pcm)>24000*600:
        raise ValueError("Voice clip longer than the 10-minute safety limit")
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing synthesized voice")
    sf.write(str(output),pcm,24000,subtype="PCM_16")
    return {"path":str(output),
            "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
            "sample_rate":24000,"duration_seconds":round(len(pcm)/24000,3),
            "review_status":"unreviewed"}

def render_jobs(jobs:list[dict],output_dir:Path,*,dry_run:bool=False)->dict:
    if not jobs:
        raise ValueError("No explicit Work tts lines found; nothing to generate")
    plan=[]
    results=[]
    for item in jobs:
        name=item.get("scene") or "SAMPLE"
        filename=f"{name}_{item['speaker']}_{item['voice']}.wav"
        output=output_dir/filename
        plan.append({"scene":name,"speaker":item["speaker"],
                     "voice":item["voice"],"language":item["language"],
                     "text":item["text"],"output":str(output)})
        if not dry_run:
            result=synthesize(item["text"],item["voice"],item["language"],output)
            result.update({"scene":name,"speaker":item["speaker"],"voice":item["voice"]})
            results.append(result)
    if dry_run:
        return {"dry_run":True,"jobs":plan,"actual_audio_clips":0}
    manifest={"schema":1,"clips":results,"human_voice_approval":False,
              "note":"Preview-only Kokoro synthesis. Copy and rights-review into assets/ for production."}
    (output_dir/"kokoro_manifest.json").write_text(
         json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"dry_run":False,"clips":results,"manifest":str(output_dir/"kokoro_manifest.json")}

def main():
    p=argparse.ArgumentParser(description="Kokoro English voice drafts; never writes screenplay")
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",help="Explicit test sentence, not part of any story")
    group.add_argument("--episode",type=Path,help="Work-prepared scene JSON with tts instructions")
    p.add_argument("--speaker",default="Mira")
    p.add_argument("--output-dir",type=Path,default=Path("/kaggle/working/two_worlds_voice"))
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    cast=load_cast()
    if args.text is not None:
        if args.speaker not in cast["speakers"]:
            p.error("Unknown speaker profile")
        if not 1<=len(args.text.strip())<=550:
            p.error("Test text must be 1..550 characters")
        profile=cast["speakers"][args.speaker]
        jobs=[{"scene":"SAMPLE","speaker":args.speaker,"text":args.text.strip(),
               "voice":profile["voice"],"language":profile["language"]}]
    else:
        ep=json.loads(args.episode.read_text(encoding="utf-8"))
        jobs=jobs_from_episode(ep,cast)
    print(json.dumps(render_jobs(jobs,args.output_dir,dry_run=args.dry_run),
                     ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
