"""Finish genuine EB01 from independently verified neural micro-shots, not stills.

S06 = first 60 clean frames from original S06 Wan v1 + 29 new genuine
model-generated S06 close-up frames. S02 = 41+41+43 new genuine model frames.
All video frames are model-generated; no optical flow, repetition, stills,
frame-hallucination, forged source receipts, or social publishing.

Every model source + SHA + original Comfy receipt is kept in a sidecar lineage.
Cutting between truly separate camera takes is a cinematic editorial choice.
The result is an OWNER REVIEW MP4, never silently art-approved.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess

from studio.eb01_v7_storyboard import SHOTS,SRC_SHA,VOICE_SHA,sha
from studio.eb01_motion_artifact_gate import inspect_clip
from studio.comfy_neural import _probe
from studio.eb01_v7_cpu_final_recovery import produce_review,OLD_GENUINE_SHA

S06_ORIGINAL_VARIANT1_SHA="2c4c5eb1b98da985d6a8cecb375710e3d3e42802a6e7f79236165a8fbf2d363f"
S05_GENUINE_SHA="806b46998911aa964398a338f14a850d55a88c9efd810ae5516ff0066f74bc8e"
S03_GENUINE_SHA="ea04508811da91b88e4ae89e7e1b704f4f8e1af63d29673df42f01d1d321bf80"
SOURCE_SEGMENT_COUNTS={"S02":(("S02_A41",41,41),("S02_B41",41,41),("S02_C45",45,43)),
                       "S06":(("S06_V1_PREFIX",93,60),("S06_TAIL29",29,29))}
CANONICAL_FRAMES={row[0]:row[2] for row in SHOTS}

class NativeFilmBlocked(RuntimeError):
    pass

def require(cond:bool,message:str):
    if not cond:raise NativeFilmBlocked(message)

def cmd(args,timeout=240):
    r=subprocess.run([str(x) for x in args],capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise NativeFilmBlocked("FFmpeg fail: "+r.stderr[-400:])
    return r.stdout

def genuine_source(name:str,src:Path,receipt:Path,frames:int,
                   reference_sha:str,known_sha:str|None=None)->dict:
    require(src.is_file() and receipt.is_file(),"Missing actual Wan native "+name)
    proven=json.loads(receipt.read_text(encoding="utf-8"))
    digest=sha(src)
    if known_sha:require(digest==known_sha,"Immutable original SHA changed "+name)
    require(proven.get("engine")=="neural_comfy" and
            proven.get("model")=="wan22-5b" and
            proven.get("inference_performed") is True and
            proven.get("published") is False and
            proven.get("clip_sha256")==digest and
            proven.get("reference_sha256")==reference_sha,
            "Native Comfy proof or canonical actor source mismatch "+name)
    meta=_probe(src)
    require(meta.get("fps")=="24" and meta.get("frames")==frames,
            "Unexpected original native frame count "+name)
    cmd(["ffmpeg","-v","error","-xerror","-i",src,"-f","null","-"],timeout=100)
    return {"name":name,"original_sha256":digest,"original_frames":frames,
            "source_receipt_sha256":sha(receipt),
            "actor_reference_sha256":reference_sha,
            "model":"wan22-5b","inference_verified":True}

def trimmed_segments(shot:str,segments:list[dict],
                     output_dir:Path,reference_sha:str)->dict:
    require(not output_dir.exists(),"Never overwrite existing scene assembly "+shot)
    output_dir.mkdir(parents=True)
    require(tuple((x["name"],x["original_frames"],x["used_frames"]) for x in segments)
            ==SOURCE_SEGMENT_COUNTS[shot],"Wrong fixed model segment plan")
    expected=CANONICAL_FRAMES[shot]
    require(sum(s["used_frames"] for s in segments)==expected,
            "Concatenated native frame count differs from approved storyboard")
    inputs=[]
    manifest=[]
    for s in segments:
        source=Path(s["source"])
        proof=Path(s["receipt"])
        name=s["name"]
        evidence=genuine_source(name,source,proof,s["original_frames"],
                                reference_sha,s.get("expected_sha"))
        used=s["used_frames"]
        require(used<=s["original_frames"],"Cannot invent movie frames")
        # Particularly original S06 v1: only first 60 genuine model frames
        # precede frame-65 catastrophic RGB corruption. Reject any artifact
        # in the explicitly kept source window, never blanket inpaint.
        prefix=output_dir/("source_"+name+".review.mp4")
        cmd(["ffmpeg","-hide_banner","-v","error","-xerror","-i",source,
             "-vf",f"trim=end_frame={used},setpts=PTS-STARTPTS",
             "-an","-r","24","-c:v","libx264","-preset","medium",
             "-crf","17","-pix_fmt","yuv420p",prefix],timeout=100)
        meta=_probe(prefix)
        gate=inspect_clip(prefix)
        require(meta.get("frames")==used and gate.get("status")=="NO_NEON_ARTIFACT_FOUND"
                and gate.get("decoded_frames")==used,
                "Corruption INSIDE selected genuine Wan subshot "+name+
                ": "+gate.get("status","unknown"))
        # Sidecar copies use descriptive names, not S02.mp4/S06.mp4,
        # so the release-stage single-final-source selector remains strict.
        native=output_dir/("native_"+name+".mp4")
        native_proof=output_dir/("native_"+name+".receipt.json")
        shutil.copy2(source,native)
        shutil.copy2(proof,native_proof)
        prefix.unlink()
        manifest.append({**evidence,"source_file":native.name,
                         "model_receipt_file":native_proof.name,
                         "used_source_start_frame":0,"used_frames":used,
                         "kept_true_model_frames_only":True})
        inputs.append(native)
    # Filter graph performs only hard cuts and exact native-frame trims.
    clips=[]
    for i,s in enumerate(segments):
        clips.append(f"[{i}:v]trim=start_frame=0:end_frame={s['used_frames']},"
                     f"setpts=PTS-STARTPTS[v{i}]")
    filters=";".join(clips)+";"+"".join(f"[v{i}]" for i in range(len(inputs)))+(
        f"concat=n={len(inputs)}:v=1:a=0[v]")
    out=output_dir/(shot+".mp4")
    cmd(["ffmpeg","-hide_banner","-v","error","-xerror",
         *sum((["-i",str(f)] for f in inputs),[]),
         "-filter_complex",filters,"-map","[v]","-an","-r","24",
         "-c:v","libx264","-preset","medium","-crf","17",
         "-pix_fmt","yuv420p","-movflags","+faststart",out],timeout=400)
    meta=_probe(out)
    scan=inspect_clip(out)
    require(meta.get("frames")==expected and meta.get("fps")=="24"
            and scan.get("decoded_frames")==expected
            and scan.get("status")=="NO_NEON_ARTIFACT_FOUND",
            "Final combined true neural scene corrupt "+shot+
            " "+scan.get("status","unknown"))
    digest=sha(out)
    # Keep genuine-model proof, but *clearly mark composition* and all
    # original model source references, never claim one continuous inference.
    first=json.loads((output_dir/manifest[0]["model_receipt_file"]).read_text())
    derived={**first,"clip_sha256":digest,"video":meta,
             "human_approved":False,"published":False,
             "derivation":{"type":"HARD_CUT_ONLY_MULTIPLE_GENUINE_WAN_I2V",
                           "original_model_sources":manifest,
                           "native_used_frames":expected,
                           "no_synthetic_or_repeated_frames":True,
                           "independent_source_count":len(inputs),
                           "human_artistic_approved":False}}
    (output_dir/(shot+".receipt.json")).write_text(json.dumps(
        derived,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    job={"schema":7,"status":"ONE_REAL_WAN_SHOT_TECHNICALLY_VERIFIED",
         "success":True,"shot":shot,"model":"wan22-5b",
         "clip_sha256":digest,"frame_count":expected,
         "neural_inference_performed":True,
         "original_reference_sha256":reference_sha,
         "temporal_rgb_artifact_gate":"NO_NEON_ARTIFACT_FOUND",
         "native_editorial_composition":True,
         "source_segments":manifest,"published":False,
         "human_artistic_approved":False}
    (output_dir/"EB01_V7_SINGLE_SHOT_JOB.json").write_text(
        json.dumps(job,ensure_ascii=False,indent=2)+"\n")
    (output_dir/"EB01_V7_NATIVE_MULTISHOT_MANIFEST.json").write_text(
        json.dumps({"scene":shot,"final_sha256":digest,
                    "source_segments":manifest,"native_frames":expected,
                    "independent_model_sources":len(inputs),
                    "published":False},indent=2)+"\n")
    return {"shot":shot,"full_scene_native_frames":expected,
            "final_sha256":digest,"source_segments":len(inputs),
            "human_artistic_review_required":True}

def collect_and_finish(dataset:Path,source_clips:dict, 
                       s06_original:Path,s06_tail:Path,
                       s02_segments:dict[str,Path],result_dir:Path)->dict:
    require(not result_dir.exists(),"Never overwrite existing EB01 film output")
    result_dir.mkdir(parents=True)
    # Read verified private canonical reference hashes without showing them
    # or the original portraits on a public CI artifact.
    from studio.eb01_v7_storyboard import gather_private
    source_lock=gather_private(dataset,result_dir/"canonical")
    require(source_lock["source_sha256"]==SRC_SHA and
            source_lock["voice_sha256"]==VOICE_SHA,
            "Wrong private original photos or six approved WAV tracks")
    reference={sid:row["reference_sha256"]
               for sid,row in source_lock["scenes"].items()}
    def component(root:Path,shot:str,meta_name:str|None=None)->tuple[Path,Path]:
        movies=list(root.rglob(shot+".mp4"))
        proofs=list(root.rglob(shot+".receipt.json"))
        require(len(movies)==1 and len(proofs)==1,"Missing/duplicate true actor MP4 "+shot)
        return movies[0],proofs[0]
    for shot,expected_sha in {"S00":OLD_GENUINE_SHA["S00"],
                              "S01":OLD_GENUINE_SHA["S01"],
                              "S03":S03_GENUINE_SHA,
                              "S05":S05_GENUINE_SHA}.items():
        movie,proof=component(source_clips[shot],shot)
        genuine_source(shot,movie,proof,CANONICAL_FRAMES[shot],
                       reference[shot],expected_sha)
    # S04 is also a genuinely checked separate isolated Wan model output.
    movie,proof=component(source_clips["S04"],"S04")
    genuine_source("S04",movie,proof,CANONICAL_FRAMES["S04"],reference["S04"])
    s06raw=s06_original/"S06.original_wan_raw.mp4"
    s06receipt=s06_original/"S06.original_wan_raw.receipt.json"
    s06tailclip,s06tailreceipt=component(s06_tail,"S06")
    s06list=[{"name":"S06_V1_PREFIX","source":s06raw,"receipt":s06receipt,
              "original_frames":93,"used_frames":60,
              "expected_sha":S06_ORIGINAL_VARIANT1_SHA},
             {"name":"S06_TAIL29","source":s06tailclip,"receipt":s06tailreceipt,
              "original_frames":29,"used_frames":29}]
    s06=trimmed_segments("S06",s06list,result_dir/"S06_RESTORED_NATIVE",reference["S06"])
    s02list=[]
    for name,source_count,kept in SOURCE_SEGMENT_COUNTS["S02"]:
        movie,proof=component(s02_segments[name],"S02")
        s02list.append({"name":name,"source":movie,"receipt":proof,
                        "original_frames":source_count,"used_frames":kept})
    s02=trimmed_segments("S02",s02list,result_dir/"S02_REAL_NATIVE_3TAKE",reference["S02"])
    shots={**source_clips,"S02":result_dir/"S02_REAL_NATIVE_3TAKE",
           "S06":result_dir/"S06_RESTORED_NATIVE"}
    movie=result_dir/"ELIZABETH_BONES_E01_REAL_NEURAL_RUSSIAN_MASTER.mp4"
    review=produce_review(dataset,shots,result_dir/"studio_master",movie)
    require(movie.is_file() and movie.stat().st_size>100000
            and review.get("dialogue_lines")==6
            and review.get("srt_cues")==6
            and review.get("all_frames_source_genuine") is True,
            "Genuine 7-take 6-Russian-voice master missing")
    status={"schema":7,"status":"FULL_7_SCENE_TRUE_WAN_RUSSIAN_MP4_GENERATED_REVIEW_REQUIRED",
            "video":movie.name,"video_sha256":sha(movie),"full_episode":True,
            "scenes":7,"russian_voice_lines":6,"captions":6,
            "S02_true_neural_source_segments":s02,
            "S06_true_neural_source_segments":s06,
            "human_artistic_review_completed":False,"published":False}
    (result_dir/"EB01_TRUE_MOVIE_QA.json").write_text(json.dumps(
        status,ensure_ascii=False,indent=2)+"\n")
    return status
