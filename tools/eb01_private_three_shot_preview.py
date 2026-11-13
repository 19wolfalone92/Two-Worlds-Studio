"""Read-only three-shot genuine Wan proof, with original Qwen Russian dialogue.

This is an *incomplete* video sampler for E01 S00-S02, NEVER the full film.
Only original private Kaggle media, checked by SHA and FFmpeg decode, may
appear in the preview. No synthetic test frames, no original source PNG,
no upload of private WAVs or original character art, no GPU inference.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from studio.eb01_v7_storyboard import DIALOGUES, VOICE_SECONDS, VOICE_SHA

OLD_SHA = {
 "S00": "7ac34976cfc5086ad068bffd837893a3c41ee6a5d48cc1fcd66f50c368e8b208",
 "S01": "fe0e45802d6df19af951dc1ebb8cfaeb396fdcce59d860d5826a62a7e7d36c87",
 "S02": "53ce7cf5245b194c9c6a47489db48acb957b552a687b5f05fa28fb62cc41ce4e",
}
SHOT_FRAMES = {"S00":49,"S01":61,"S02":125}
MODEL = "wan22-5b"


class PreviewBlocked(ValueError): pass


def sha(p:Path)->str:
    d=hashlib.sha256()
    with p.open("rb") as fd:
        for block in iter(lambda:fd.read(1024*1024),b""):
            d.update(block)
    return d.hexdigest()


def run(args,timeout=200):
    result=subprocess.run(args,check=False,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise PreviewBlocked("FFmpeg pipeline failed: "+result.stderr[-900:])
    return result


def probe(movie:Path)->dict:
    return json.loads(run([
        "ffprobe","-hide_banner","-v","error","-count_frames",
        "-show_entries","stream=codec_type,codec_name,width,height,avg_frame_rate,nb_read_frames",
        "-of","json",str(movie)],timeout=100).stdout)


def verify_private_sources(original:Path,s02:Path,voices:Path)->dict:
    result={}
    for sid,expected in OLD_SHA.items():
        root=s02 if sid=="S02" else original
        movies=list(root.rglob(sid+".mp4"))
        proofs=list(root.rglob(sid+".receipt.json"))
        if len(movies)!=1 or len(proofs)!=1:
            raise PreviewBlocked(f"Missing/ambiguous genuine private Wan {sid} model movie or receipt")
        movie,proof=movies[0],proofs[0]
        if sha(movie)!=expected:
            raise PreviewBlocked(f"Original private model MP4 SHA mismatch: {sid}")
        model=json.loads(proof.read_text(encoding="utf-8"))
        if (model.get("model")!=MODEL or model.get("engine")!="neural_comfy"
                or model.get("inference_performed") is not True
                or model.get("clip_sha256")!=expected
                or model.get("published") is not False):
            raise PreviewBlocked(f"Missing original genuine model receipt: {sid}")
        meta=probe(movie)
        video=[row for row in meta["streams"] if row.get("codec_type")=="video"]
        if (len(video)!=1 or video[0].get("codec_name")!="h264"
                or video[0].get("width")!=320 or video[0].get("height")!=576
                or video[0].get("avg_frame_rate")!="24/1"
                or int(video[0].get("nb_read_frames",0))!=SHOT_FRAMES[sid]):
            raise PreviewBlocked(f"Actual native model frame count/codec mismatch: {sid}")
        run(["ffmpeg","-hide_banner","-nostdin","-v","error","-xerror",
             "-i",str(movie),"-f","null","-"],timeout=180)
        result[sid]={"path":movie,"sha256":expected,"frames":SHOT_FRAMES[sid]}
    for sid in ("S01","S02"):
        wav=list(voices.rglob(sid+".wav"))
        if len(wav)!=1 or sha(wav[0])!=VOICE_SHA[sid]:
            raise PreviewBlocked(f"Missing original Russian Qwen WAV: {sid}")
        result[sid]["voice"]=wav[0]
    return result


def srt_time(t:float)->str:
    millis=round(t*1000)
    h,millis=divmod(millis,3600000)
    m,millis=divmod(millis,60000)
    s,ms=divmod(millis,1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def produce_preview(original:Path,s02:Path,voices:Path,out:Path)->dict:
    sources=verify_private_sources(original,s02,voices)
    if out.exists() or out.is_symlink():
        raise PreviewBlocked("Never overwrite user or original output")
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        video_paths=[]
        cues=[]
        clock=0.0
        for sid in SHOT_FRAMES:
            frames=SHOT_FRAMES[sid]
            length=frames/24
            inputs=["-i",str(sources[sid]["path"])]
            if sid in ("S01","S02"):
                inputs+=["-i",str(sources[sid]["voice"])]
                offset_ms=120
                audio=f"[1:a]aresample=48000,adelay={offset_ms},apad,atrim=duration={length},asetpts=PTS-STARTPTS[a]"
                cues.append((clock+.12,clock+.12+VOICE_SECONDS[sid],DIALOGUES[sid]))
            else:
                inputs+=["-f","lavfi","-i","anullsrc=r=48000:cl=mono"]
                audio=f"[1:a]atrim=duration={length},asetpts=PTS-STARTPTS[a]"
            graph=f"[0:v]fps=24,scale=576:1024:flags=lanczos,setsar=1,format=yuv420p[v];"+audio
            seg=root/(sid+".mp4")
            run([
                "ffmpeg","-hide_banner","-nostdin","-v","error","-y",
                *inputs,"-filter_complex",graph,
                "-map","[v]","-map","[a]",
                "-frames:v",str(frames),
                "-c:v","libx264","-preset","medium","-crf","19",
                "-pix_fmt","yuv420p","-r","24",
                "-c:a","aac","-b:a","160k",
                "-movflags","+faststart",str(seg),
            ],timeout=220)
            video_paths.append(seg)
            clock+=length
        concat=root/"playlist.txt"
        concat.write_text("".join("file '"+str(p)+"'\n" for p in video_paths),encoding="utf-8")
        run(["ffmpeg","-hide_banner","-nostdin","-v","error","-y",
             "-f","concat","-safe","0","-i",str(concat),
             "-c","copy","-movflags","+faststart",str(out)],timeout=200)
        # Exact two independently generated, authentic Russian voice clips.
        srt=out.with_suffix(".srt")
        srt.write_text("".join(
            f"{n}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n\n"
            for n,(start,end,text) in enumerate(cues,1)
        ),encoding="utf-8")
    stream=probe(out)
    v=[row for row in stream["streams"] if row.get("codec_type")=="video"]
    a=[row for row in stream["streams"] if row.get("codec_type")=="audio"]
    if (len(v)!=1 or len(a)!=1 or v[0]["width"]!=576 or v[0]["height"]!=1024
            or v[0]["avg_frame_rate"]!="24/1"
            or int(v[0].get("nb_read_frames",0))!=sum(SHOT_FRAMES.values())):
        raise PreviewBlocked("Review MP4 contains missing real video frames or audio")
    run(["ffmpeg","-hide_banner","-nostdin","-v","error","-xerror",
         "-i",str(out),"-f","null","-"],timeout=220)
    report={
        "schema":7,
        "status":"REAL_WAN_3_OF_7_PREVIEW_NOT_FULL_FILM",
        "source_shots":{s:{"sha256":row["sha256"],"frames":row["frames"]}
                        for s,row in sources.items()},
        "actual_video_shots":3,
        "remaining_unmade_shots":["S03","S04","S05","S06"],
        "original_russian_Qwen_WAV_lines":2,
        "subtitles":len(cues),
        "review_mp4_name":out.name,
        "review_mp4_sha256":sha(out),
        "review_srt_sha256":sha(out.with_suffix(".srt")),
        "review_frame_count":int(v[0]["nb_read_frames"]),
        "review_has_audio":True,
        "full_episode_ready":False,"human_artistic_approved":False,
        "private_character_pngs_or_wavs_in_artifact":False,
        "published_to_social":False
    }
    out.with_suffix(".qa.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return report


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--s02",type=Path,required=True)
    p.add_argument("--voices",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    x=p.parse_args()
    print(json.dumps(produce_preview(x.original,x.s02,x.voices,x.output),
                     ensure_ascii=False,indent=2))
