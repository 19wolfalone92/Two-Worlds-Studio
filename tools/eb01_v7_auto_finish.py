"""One owner-authorized, sequential, genuine Wan2.2 completion runner for EB01.

No mock/still fallback. No social publication. Never overwrite original media.
Wait for the *already running* S04, verify its actual private MP4, then
generate missing S05 and S06, replace rejected neon-streaked S02 with one
independent quality-retake seed, and assemble ALL seven original native shots
and six original Qwen Russian voices using TWO WORLDS STUDIO.

This may take hours and may fail an artistic gate: do not call a green
GPU SUBMISSION a finished film. GitHub Actions must own all accountability.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from studio.eb01_v7_storyboard import SHOTS, sha
from studio.eb01_motion_artifact_gate import inspect_clip
from studio.comfy_neural import _probe
from studio.eb01_v7_cpu_final_recovery import produce_review, OLD_GENUINE_SHA
from tools.kaggle_eb01_v7_single_shot import check_configuration

REPO="19wolfalone92/Two-Worlds-Studio"
BRANCH="fix/eb01-s02-private-raw-preservation-20261113"
SOURCE_DATASET="wolf19loner92/eb01-v7-private-production-sources"
ORIGINAL="wolf19loner92/eb01-v7-native-full-c188f4380731"
S03="wolf19loner92/eb01-v7-s03-only-1e7b0ac41e3a"
S04="wolf19loner92/eb01-v7-s04-only-2333bba16751"
MAX_POLL_SECONDS=100*60

def run(*args, timeout=240, **kwargs):
    cmd=[str(v) for v in args]
    p=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout,**kwargs)
    if p.returncode:raise RuntimeError(
        "Command failed: "+cmd[0]+" "+(" ".join(cmd[1:3]))+
        " "+p.stdout[-1000:]+" "+p.stderr[-1000:])
    return p.stdout+p.stderr

def record(path:Path,data:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

def kernel_status(kernel:str)->str:
    out=run("kaggle","kernels","status",kernel,timeout=100)
    states=re.findall(r"KernelWorkerStatus\.([A-Z_]+)",out)
    if len(states)!=1:raise RuntimeError("Ambiguous private Kaggle status: "+out[-350:])
    return states[0]

def await_existing(kernel:str,shot:str,statusfile:Path,deadline:int=MAX_POLL_SECONDS):
    start=time.monotonic()
    while True:
        state=kernel_status(kernel)
        record(statusfile,{"current_shot":shot,"kernel":kernel,
                           "status":state,"finished_mp4":False,
                           "new_gpu_auto_retries":False,
                           "published":False})
        print("WAN_KAGGLE_ACTUAL_STATUS",shot,state,flush=True)
        if state=="COMPLETE":return
        if state in ("ERROR","CANCEL_ACKNOWLEDGED","CANCELLED","FAILED"):
            raise RuntimeError("Existing genuine GPU inference failed "+shot+" "+state)
        if time.monotonic()-start>deadline:
            raise RuntimeError("Kaggle job did not complete within bounded GPU wait: "+shot)
        time.sleep(90)

def download_kernel(kernel:str,dest:Path):
    dest.mkdir(parents=True,exist_ok=False)
    run("kaggle","kernels","output",kernel,"-p",dest,"--force",timeout=240)

def verify_shot(shot:str,folder:Path,known_sha:str|None=None)->dict:
    movies=list(folder.rglob(shot+".mp4"))
    jobs=list(folder.rglob("EB01_V7_SINGLE_SHOT_JOB.json"))
    receipts=list(folder.rglob(shot+".receipt.json"))
    if len(movies)!=1 or len(jobs)!=1 or len(receipts)!=1:
        raise RuntimeError("Genuine native Wan MP4+receipts absent/ambiguous "+shot)
    movie=movies[0]
    digest=sha(movie)
    data=json.loads(jobs[0].read_text())
    proof=json.loads(receipts[0].read_text())
    requested=next(row[2] for row in SHOTS if row[0]==shot)
    if known_sha and digest!=known_sha:
        raise RuntimeError("Previously verified source changed! "+shot)
    if (data.get("status")!="ONE_REAL_WAN_SHOT_TECHNICALLY_VERIFIED"
        or data.get("success") is not True or data.get("shot")!=shot
        or data.get("clip_sha256")!=digest
        or data.get("frame_count")!=requested
        or data.get("model")!="wan22-5b"
        or data.get("neural_inference_performed") is not True
        or proof.get("engine")!="neural_comfy"
        or proof.get("model")!="wan22-5b"
        or proof.get("inference_performed") is not True
        or proof.get("clip_sha256")!=digest):
        raise RuntimeError("Private MP4 model provenance failed "+shot)
    video=_probe(movie)
    if video.get("frames")!=requested or video.get("fps")!="24":
        raise RuntimeError("Wan video frames/FPS wrong "+shot)
    analysis=inspect_clip(movie)
    if (analysis["decoded_frames"]!=requested or
        analysis["status"]!="NO_NEON_ARTIFACT_FOUND"):
        raise RuntimeError("NEON/RAINBOW RGB visual contamination "+shot+
                           ": "+analysis["status"]+
                           " flagged "+str(analysis.get("flagged_frames",[])[:30]))
    run("ffmpeg","-v","error","-xerror","-i",movie,
        "-f","null","-",timeout=180)
    return {"scene":shot,"kernel":data.get("nonce"),"mp4_sha256":digest,
            "frames":requested,"true_neural":True,
            "artificial_frames":0,"technical_gate":"PASS",
            "human_artistic_approved":False}

def issue_single(shot:str,root:Path,revision:str,nonce:str,retake=False):
    assert re.fullmatch("[a-f0-9]{40}",revision)
    cfg={"schema":7,"nonce":nonce,"shot":shot,"model":"wan22-5b",
         "private":True,"new_gpu_jobs":1,
         "source_dataset":SOURCE_DATASET,
         "revision":revision,"branch":BRANCH,
         "allow_social_publish":False,
         "allow_regenerate_good_shots":False,
         "quality_retake":retake}
    assert check_configuration(cfg)==shot
    body=Path("tools/kaggle_eb01_v7_single_shot.py").read_text()
    placeholder="__EB01_V7_RESUME_SHOT_CONFIG_B64__"
    if body.count(placeholder)!=1:raise RuntimeError("Invalid worker config placeholder")
    source=body.replace(placeholder,base64.b64encode(json.dumps(cfg,sort_keys=True).encode()).decode())
    if len(source)>70000:raise RuntimeError("Unexpected worker source size")
    slug="eb01-v7-"+shot.lower()+"-only-"+hashlib.sha256(nonce.encode()).hexdigest()[:12]
    kernel="wolf19loner92/"+slug
    folder=root/("launch_"+shot)
    folder.mkdir()
    (folder/"kernel_main.py").write_text(source)
    meta={"id":kernel,"title":slug,"code_file":"kernel_main.py",
          "language":"python","kernel_type":"script","is_private":"true",
          "enable_gpu":"true","enable_internet":"true",
          "machine_shape":"NvidiaTeslaT4","dataset_sources":[SOURCE_DATASET],
          "competition_sources":[],"kernel_sources":[],"model_sources":[]}
    record(folder/"kernel-metadata.json",meta)
    run("kaggle","kernels","push","-p",folder,
        "--timeout","3600","--accelerator","NvidiaTeslaT4",timeout=420)
    print("GENUINE_PRIVATE_GPU_SCENE_SUBMITTED",shot,kernel,flush=True)
    return kernel

def old_s00_s01(path:Path):
    found={}
    for sid,expected in OLD_GENUINE_SHA.items():
        vids=list(path.rglob(sid+".mp4"))
        proofs=list(path.rglob(sid+".receipt.json"))
        if len(vids)!=1 or len(proofs)!=1 or sha(vids[0])!=expected:
            raise RuntimeError("Original verified S00/S01 missing or changed "+sid)
        result=_probe(vids[0])
        expected_frames=next(row[2] for row in SHOTS if row[0]==sid)
        if result.get("frames")!=expected_frames:
            raise RuntimeError("Original old real scene frames wrong "+sid)
        run("ffmpeg","-v","error","-xerror","-i",vids[0],"-f","null","-",timeout=100)
        found[sid]={"sha256":expected,"frames":expected_frames}
    return found

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--revision",required=True)
    a.add_argument("--run-id",required=True)
    args=a.parse_args()
    if os.environ.get("KAGGLE_API_TOKEN","")=="":
        raise RuntimeError("Kaggle GitHub secret unavailable")
    if not re.fullmatch(r"[0-9]+",args.run_id):
        raise RuntimeError("Invalid Github run ID")
    dest=args.output
    dest.mkdir(parents=True,exist_ok=False)
    report={"schema":7,"status":"RUNNING_REAL_7_TAKE_PRODUCTION",
            "genuine_finished_shots":["S00","S01","S03"],
            "published":False,"human_visual_reviewed":False,
            "no_synthetic_fallback":True,"six_original_ru_voices_required":True,
            "sources":{},"pending":["S04","S05","S06","S02_QUALITY_RETAKE"]}
    record(dest/"EB01_V7_STATUS.json",report)
    with tempfile.TemporaryDirectory(prefix="eb01-real-neural-") as temp:
        root=Path(temp)
        try:
            # The already-running S04 is NEVER resubmitted.
            sources={"S00":root/"original",
                     "S01":root/"original"}
            shots=[("S03",S03,"ea04508811da91b88e4ae89e7e1b704f4f8e1af63d29673df42f01d1d321bf80"),
                   ("S04",S04,None)]
            for sid,kernel,known in shots:
                await_existing(kernel,sid,dest/"EB01_V7_STATUS.json")
                folder=root/sid
                download_kernel(kernel,folder)
                report["sources"][sid]=verify_shot(sid,folder,known)
                sources[sid]=folder
                report["genuine_finished_shots"].append(sid)
                report["pending"].remove(sid)
                record(dest/"EB01_V7_STATUS.json",report)
            # All remaining clips are completely independent: each verified
            # BEFORE investment in the next GPU shot. No silent retries.
            for sid in ("S05","S06","S02"):
                retake=(sid=="S02")
                nonce=f"eb01v7-full-owner-{args.run_id}-{sid.lower()}"+("-retake" if retake else "")
                kernel=issue_single(sid,root,args.revision,nonce,retake=retake)
                await_existing(kernel,sid,dest/"EB01_V7_STATUS.json")
                folder=root/sid
                download_kernel(kernel,folder)
                report["sources"][sid]=verify_shot(sid,folder)
                report["sources"][sid]["private_kernel"]=kernel
                sources[sid]=folder
                report["genuine_finished_shots"].append(sid)
                report["pending"].remove("S02_QUALITY_RETAKE" if retake else sid)
                record(dest/"EB01_V7_STATUS.json",report)
            download_kernel(ORIGINAL,root/"original")
            report["sources"].update(old_s00_s01(root/"original"))
            # Exactly 7 genuine native clips, including *retake* S02, plus 6 RU voices.
            dataset=root/"private_dataset"
            dataset.mkdir()
            run("kaggle","datasets","download","-d",SOURCE_DATASET,
                "-p",dataset,"--unzip",timeout=480)
            movie=dest/"ELIZABETH_BONES_E01_FULL_REAL_WAN22_RUSSIAN_REVIEW.mp4"
            qa=produce_review(dataset,sources,root/"studio_master",movie)
            if qa.get("dialogue_lines")!=6 or qa.get("srt_cues")!=6 or qa.get("all_frames_source_genuine") is not True:
                raise RuntimeError("Six Russian voice/7 genuine scenes release contract failed")
            report.update({"status":"FULL_NEURAL_VIDEO_CREATED_ARTISTIC_REVIEW_REQUIRED",
                           "published":False,"human_visual_reviewed":False,
                           "genuine_scene_count":7,
                           "voice_lines":6,"final_video_sha256":sha(movie),
                           "final_filename":movie.name,"pending":[]})
            record(dest/"EB01_V7_STATUS.json",report)
            print("SEVEN_GENUINE_WAN_SCENES_FULL_RUSSIAN_MP4_CREATED",sha(movie),flush=True)
        except BaseException as exc:
            report["status"]="BLOCKED_NO_FALSE_FILM_CLAIM"
            report["root_cause"]=type(exc).__name__+": "+str(exc)[:1500]
            record(dest/"EB01_V7_STATUS.json",report)
            raise

if __name__=="__main__":
    main()
