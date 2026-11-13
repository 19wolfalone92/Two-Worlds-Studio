"""Create real, watchable ORIGINAL-VOICE review cut from EXISTING Wan moving footage.

NO GPU jobs, no fabricated S02 neural inference. Because independent S02
diffusion repeatedly failed, scene S02 is editorially re-cut from 3 existing
Kostya Wan moving shots. A clearly labelled reuse is preferable to another
false 7-original-shot claim or endless no-video status. All other scenes are
native, source-hashed real Wan footage. Six original SHA-locked Russian Qwen
voice tracks and six correctly timed original subtitles.

This is an owner-review EDITORIAL REUSE cut, not a final 7 independently
inferred shot movie. The report is intentionally explicit about that fact.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,tempfile,shutil
from datetime import timedelta
from pathlib import Path

from studio.eb01_v7_storyboard import (
    SHOTS,SRC_SHA,VOICE_SHA,VOICE_SECONDS,DIALOGUES,sha,gather_private)
from studio.comfy_neural import _probe
from studio.eb01_motion_artifact_gate import inspect_clip
from studio.eb01_v7_cpu_final_recovery import OLD_GENUINE_SHA
from studio.eb01_v7_native_micro_master import (
    genuine_source, trimmed_segments,
    S03_GENUINE_SHA,S05_GENUINE_SHA,S06_ORIGINAL_VARIANT1_SHA)
from tools.kaggle_eb01_v7_full_entry import create_score

DATASET="wolf19loner92/eb01-v7-private-production-sources"
SOURCES={
 "ORIGINAL":"wolf19loner92/eb01-v7-native-full-c188f4380731",
 "S03":"wolf19loner92/eb01-v7-s03-only-1e7b0ac41e3a",
 "S04":"wolf19loner92/eb01-v7-s04-only-2333bba16751",
 "S05":"wolf19loner92/eb01-v7-s05-only-75dbb6f053bf",
 "S06_ORIGINAL":"wolf19loner92/eb01-v7-s06-only-beca33b6874e",
 "S06_TAIL":"wolf19loner92/eb01-v7-s06-tail29-17f5716049d7",
}
# Exactly 29+66+30=125 real actor frames. 3 separate camera/acting takes.
# Every frame originally computed by Wan2.2, but S02 is reused footage;
# DO NOT claim a separate original S02 AI inference.
REUSE=(("S00",20,49),("S04",0,66),("S06",0,30))
FPS=24
FRAME_COUNTS={row[0]:row[2] for row in SHOTS}
FRAMES_TOTAL=sum(FRAME_COUNTS.values())

def insist(ok,message):
    if not ok: raise RuntimeError(message)

def cmd(args,timeout=300):
    p=subprocess.run([str(x) for x in args],capture_output=True,text=True,timeout=timeout)
    if p.returncode: raise RuntimeError("Real media command failed "+str(args[:3])+" "+p.stderr[-900:])
    return p.stdout

def one(root:Path,filename:str)->Path:
    found=list(root.rglob(filename))
    insist(len(found)==1,"Missing or multiple existing REAL source: "+filename)
    return found[0]

def fetch_private(kernel:str,dest:Path)->Path:
    dest.mkdir(parents=True,exist_ok=False)
    cmd(["kaggle","kernels","output",kernel,"-p",dest,"--force"],timeout=360)
    return dest

def seconds_to_srt(t:float)->str:
    ms=round(t*1000)
    h,rem=divmod(ms,3600000)
    minute,rem=divmod(rem,60000)
    sec,millis=divmod(rem,1000)
    return f"{h:02d}:{minute:02d}:{sec:02d},{millis:03d}"

def stage_sources(work:Path)->tuple[Path,dict,dict]:
    private=work/"dataset"
    private.mkdir()
    cmd(["kaggle","datasets","download","-d",DATASET,
         "-p",private,"--unzip"],timeout=480)
    studio=work/"studio"
    lock=gather_private(private,studio)
    insist(lock["source_sha256"]==SRC_SHA and lock["voice_sha256"]==VOICE_SHA,
           "Original protected portraits/six Qwen voices have changed")
    fetched={name:fetch_private(kernel,work/name) for name,kernel in SOURCES.items()}
    clips=studio/"clips"
    clips.mkdir()
    provenance={}
    for sid,source_key,expected_sha in (
        ("S00","ORIGINAL",OLD_GENUINE_SHA["S00"]),
        ("S01","ORIGINAL",OLD_GENUINE_SHA["S01"]),
        ("S03","S03",S03_GENUINE_SHA),
        ("S04","S04",None),
        ("S05","S05",S05_GENUINE_SHA)):
        raw=one(fetched[source_key],sid+".mp4")
        receipt=one(fetched[source_key],sid+".receipt.json")
        meta=genuine_source(sid,raw,receipt,FRAME_COUNTS[sid],
                            lock["scenes"][sid]["reference_sha256"],
                            expected_sha)
        qa=inspect_clip(raw)
        insist(qa["status"]=="NO_NEON_ARTIFACT_FOUND" and
               qa["decoded_frames"]==FRAME_COUNTS[sid],
               "Authentic source contains neon-corrupted model frame: "+sid)
        shutil.copy2(raw,clips/(sid+".mp4"))
        shutil.copy2(receipt,clips/(sid+".receipt.json"))
        provenance[sid]={
          "original_native_ai_inference":True,
          "source_type":"Wan2.2 5B real private MP4",
          "source_sha256":meta["original_sha256"],
          "source_receipt_sha256":meta["source_receipt_sha256"],
          "frames":FRAME_COUNTS[sid],
          "reference_sha256":lock["scenes"][sid]["reference_sha256"],
          "original_actor":next(r[1] for r in SHOTS if r[0]==sid),
        }
        print("EB01_TRUE_WAN_SOURCE_VERIFIED",sid,meta["original_sha256"],flush=True)
    # Build S06 from two separately authentic original sources, without any
    # interpolated frames or loop. Full RAW v1 source exists only privately.
    s06raw=one(fetched["S06_ORIGINAL"],"S06.original_wan_raw.mp4")
    s06proof=one(fetched["S06_ORIGINAL"],"S06.original_wan_raw.receipt.json")
    tail=one(fetched["S06_TAIL"],"S06.mp4")
    tailproof=one(fetched["S06_TAIL"],"S06.receipt.json")
    restored=trimmed_segments("S06",[
      {"name":"S06_V1_PREFIX","source":s06raw,
       "receipt":s06proof,"original_frames":93,"used_frames":60,
       "expected_sha":S06_ORIGINAL_VARIANT1_SHA},
      {"name":"S06_TAIL29","source":tail,
       "receipt":tailproof,"original_frames":29,"used_frames":29}
      ],work/"genuine_restored_S06",lock["scenes"]["S06"]["reference_sha256"])
    shutil.copy2(work/"genuine_restored_S06"/"S06.mp4",clips/"S06.mp4")
    shutil.copy2(work/"genuine_restored_S06"/"S06.receipt.json",clips/"S06.receipt.json")
    provenance["S06"]={
      "original_native_ai_inference":True,"recombined_original_model_segments":2,
      "model":"Wan2.2 5B","frames":89,"sha256":restored["final_sha256"],
      "source_segment_cut":["original_real_frames[0:60]","new_genuine_tail[0:29]"],
      "reference_sha256":lock["scenes"]["S06"]["reference_sha256"],
    }
    return studio,lock,provenance

def editorial_s02(studio:Path,provenance:dict)->dict:
    clips=studio/"clips"
    insist(sum(end-begin for _,begin,end in REUSE)==FRAME_COUNTS["S02"],
           "Wrong real actor performance cut length")
    src=[]
    source_lineage=[]
    for sid,begin,end in REUSE:
        path=clips/(sid+".mp4")
        meta=_probe(path)
        insist(meta["frames"]>=end and meta["fps"]=="24",
               "Cannot use genuine source shot "+sid)
        src.append(path)
        source_lineage.append({
          "original_scene":sid,"source_sha256":sha(path),
          "used_actual_native_frames":list(range(begin,end)),
          "real_motion_frames":end-begin,
          "reused_from_another_scene":True,
          "model_generated":True,
          "new_original_S02_inference":False})
    filters=[]
    for i,(_,a,b) in enumerate(REUSE):
        filters.append(f"[{i}:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[v{i}]")
    joined=";".join(filters)+";"+"".join(f"[v{i}]" for i in range(len(src)))+f"concat=n={len(src)}:v=1:a=0[v]"
    destination=clips/"S02.mp4"
    cmd(["ffmpeg","-hide_banner","-v","error","-xerror",
         *sum((["-i",p] for p in src),[]),
         "-filter_complex",joined,
         "-map","[v]","-an","-r","24","-c:v","libx264",
         "-preset","medium","-crf","17","-pix_fmt","yuv420p",
         "-movflags","+faststart",destination],timeout=220)
    meta=_probe(destination)
    qa=inspect_clip(destination)
    insist(meta["frames"]==125 and meta["fps"]=="24" and
           qa["decoded_frames"]==125 and
           qa["status"]=="NO_NEON_ARTIFACT_FOUND",
           "Edited S02 contains corruption or is not 125 REAL source frames")
    cmd(["ffmpeg","-v","error","-xerror","-i",destination,"-f","null","-"],timeout=120)
    receipt={"schema":7,
       "scene":"S02","source_engine":"REAL_WAN_NATIVE_EDITORIAL_REUSE",
       "true_original_s02_wan_inference":False,
       "identical_original_actor":SRC_SHA["KOSTYA"],
       "original_character":"KOSTYA",
       "source_segments":source_lineage,
       "source_segments_count":len(REUSE),
       "all_frames_originally_model_generated":True,
       "frames_repeated_within_S02":False,
       "frames_reused_elsewhere_in_film":True,
       "additional_model_generated_frames":False,
       "synthetic_interpolated_frames":False,
       "s02_clip_sha256":sha(destination),"frames":125,
       "reference_for_original_s02":False,
       "original_s02_dialogue_audio_source_sha256":VOICE_SHA["S02"],
       "human_artistic_approval":False,"published":False}
    (clips/"S02.editorial_reuse_receipt.json").write_text(
      json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    provenance["S02"]={"original_native_ai_inference":False,
      "assembled_from_actual_Wan_moving_source":True,
      "reused_frames_from_S00_S04_S06":True,
      "frames":125,"sha256":sha(destination),
      "quality_status":"OWNER_REVIEW_CUT_NOT_FINAL_ORIGINAL_S02", 
      "source_segments":source_lineage}
    print("EB01_S02_WAN_REAL_FOOTAGE_EDITORIAL_REUSE_CREATED",
          sha(destination),flush=True)
    return receipt

def produce_movie(studio:Path,dest:Path,provenance:dict,lock:dict)->dict:
    clips=studio/"clips"
    for sid,count in FRAME_COUNTS.items():
        p=clips/(sid+".mp4")
        meta=_probe(p)
        insist(meta["frames"]==count and meta["fps"]=="24",
               "Final clip missing genuine 24fps source frames "+sid)
    # Track original speech exactly; six SHA-pinned Qwen-generated Russian WAVs
    # not text-to-speech regeneration, not wrong voices or substituted samples.
    starts={}
    cursor=0
    captions=[]
    for sid,*_ in SHOTS:
        starts[sid]=cursor/FPS
        if sid in DIALOGUES:
            begin=cursor/FPS + 0.09
            duration=FRAME_COUNTS[sid]/FPS
            end=(cursor+FRAME_COUNTS[sid])/FPS-0.14
            captions.append((begin,end,DIALOGUES[sid]))
        cursor+=FRAME_COUNTS[sid]
    insist(cursor==FRAMES_TOTAL and len(captions)==6,
           "Source scene frame count/caption timing inconsistent")
    srt=dest/"ELIZABETH_BONES_E01_RU_OWNER_REVIEW.srt"
    srt.write_text("\n\n".join(
      f"{i}\n{seconds_to_srt(a)} --> {seconds_to_srt(b)}\n{line}"
      for i,(a,b,line) in enumerate(captions,1))+"\n",encoding="utf-8")
    # Native movie scenes are hard cut using a trusted concat filter rather
    # than looping/zooming a static portrait or falsely claiming 7 new jobs.
    scene_list=dest/"actual_seven_neural_scenes.ffconcat"
    scene_list.write_text("ffconcat version 1.0\n"+"".join(
        "file '"+str((clips/(row[0]+".mp4")).resolve()).replace("'","'\\''")+"'\n"
        for row in SHOTS),encoding="utf-8")
    silent=dest/"visual_studio_cut.mp4"
    cmd(["ffmpeg","-hide_banner","-v","error","-xerror",
        "-safe","0","-f","concat","-i",scene_list,
        "-vf","subtitles="+str(srt)+
           ":force_style='FontName=DejaVu Sans,FontSize=18,PrimaryColour=&H00FFFFFF,Outline=2,Shadow=1,MarginV=38'",
        "-r","24","-an","-c:v","libx264","-preset","medium",
        "-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",silent],timeout=470)
    tmpinfo=_probe(silent)
    insist(tmpinfo["frames"]==FRAMES_TOTAL and tmpinfo["fps"]=="24",
           "Studio concat did not preserve 563 original neural source frames")
    # Voice time positions are aligned to actual native source frame clocks.
    # Use source-provided WAV and generated original music (no copyrighted music).
    create_score(studio,FRAMES_TOTAL/FPS+0.5)
    wavs=[]
    delayed=[]
    for i,sid in enumerate(VOICE_SHA,1):
        wav=studio/"voices"/(sid+".wav")
        insist(sha(wav)==VOICE_SHA[sid],"Original approved Qwen voice changed "+sid)
        ms=round((starts[sid]+0.07)*1000)
        delayed.append(f"[{i}:a]aresample=async=1:first_pts=0,adelay={ms}:all=1[d{i}]")
        wavs.append(wav)
    delayed.append("[7:a]volume=0.095[bg]")
    filters=";".join(delayed)+";"+"".join(f"[d{i}]" for i in range(1,7))+(
        "[bg]amix=inputs=7:duration=longest:normalize=0,alimiter=limit=0.94[a]")
    full=dest/"ELIZABETH_BONES_SOBESEDOVANIE_FULL_RU_EDITORIAL_REVIEW.mp4"
    cmd(["ffmpeg","-hide_banner","-v","error","-xerror",
         "-i",silent,*sum((["-i",w] for w in wavs),[]),
         "-i",studio/"score.wav",
         "-filter_complex",filters,"-map","0:v:0","-map","[a]",
         "-c:v","copy","-c:a","aac","-b:a","192k",
         "-t",str(FRAMES_TOTAL/FPS),
         "-movflags","+faststart",full],timeout=270)
    info=json.loads(cmd(["ffprobe","-v","error","-count_frames","-show_streams",
              "-show_format","-of","json",full]))
    streams=info.get("streams",[])
    video=[x for x in streams if x.get("codec_type")=="video"]
    audio=[x for x in streams if x.get("codec_type")=="audio"]
    insist(len(video)==len(audio)==1 and
           int(video[0].get("nb_read_frames",0))==FRAMES_TOTAL,
           "Final owner-review film missing real video frames or original Russian audio")
    cmd(["ffmpeg","-v","error","-xerror","-i",full,"-f","null","-"],timeout=270)
    insist(srt.read_text().count("-->")==6,
           "Missing six original Russian subtitle cues")
    report={"schema":7,
      "status":"REAL_ANIMATED_FULL_RU_OWNER_REVIEW_EDITORIAL_S02_REUSE",
      "file":full.name,"sha256":sha(full),
      "duration_seconds":round(FRAMES_TOTAL/FPS,5),
      "actual_video_frames":FRAMES_TOTAL,
      "scenes_on_timeline":len(SHOTS),"russian_voice_lines":6,
      "russian_subtitle_cues":6,
      "original_Qwen_voices_SHA":VOICE_SHA,
      "actor_original_portraits_SHA":SRC_SHA,
      "all_frame_source_neural_real_Wan":True,
      "independent_original_S02_gpu_generation":False,
      "S02_temporarily_reuses_existing_Kostya_neural_footage":True,
      "reused_motion_in_S02_must_receive_owner_review":True,
      "video_codec":video[0].get("codec_name"),
      "audio_codec":audio[0].get("codec_name"),
      "original_artistic_quality_approved":False,
      "social_published":False,
      "scene_source_provenance":provenance}
    (dest/"EB01_TRUE_RU_EDITORIAL_REVIEW_QA.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("EB01_REAL_COMPLETE_OWNER_REVIEW_VIDEO_CREATED",full.name,
          report["sha256"],report["duration_seconds"],flush=True)
    # Upload only final video, subtitle and factual provenance; never
    # upload the original private character photos or six voice WAVs.
    silent.unlink()
    scene_list.unlink()
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    dest=args.output
    insist(not dest.exists(),"Do not overwrite any previous owner-review film")
    insist(bool(os.environ.get("KAGGLE_API_TOKEN")),"Cannot access original private GPU clips")
    dest.mkdir(parents=True)
    progress=dest/"EB01_REVIEW_CUT_PROGRESS.json"
    state={"status":"FETCHING_EXISTING_REAL_WAN_SCENES",
           "original_seven_independent_shots_ready":False,
           "video_ready":False,"published":False}
    progress.write_text(json.dumps(state,indent=2)+"\n")
    with tempfile.TemporaryDirectory(prefix="eb01-CPU-TRUE-FILM-") as folder:
        try:
            stage,lock,lines=stage_sources(Path(folder))
            reuse=editorial_s02(stage,lines)
            info=produce_movie(stage,dest,lines,lock)
            state.update({"status":"ACTUAL_REAL_WAN_FULL_OWNER_REVIEW_READY",
              "video_ready":True,
              "filename":info["file"],"sha256":info["sha256"],
              "reuses_original_Kostya_clips_in_S02":True,
              "original_independent_S02_gpu_scene":False})
            progress.write_text(json.dumps(state,indent=2)+"\n")
        except BaseException as error:
            state.update({"status":"BLOCKED_NO_FALSE_FILM","error_type":type(error).__name__,
                          "reason":str(error)[:900]})
            progress.write_text(json.dumps(state,indent=2)+"\n")
            raise

if __name__=="__main__":
    main()
