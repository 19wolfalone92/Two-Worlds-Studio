"""One-shot unattended real GPU + final mastering of EB01 seven neural scenes.

S06 short genuine native I2V lives in a separate queued Kaggle T4 session.
This process first checks its *actual* success. Only then generates exact
S02_A41, B41 and C45, independently checking provenance and visual QC.
No fake fallback, no automatic social publication, bounded GPU runtime,
and a truthful status artifact on ANY failure.
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
import tempfile
import time

from tools.kaggle_eb01_v7_micro_shot import SEGMENT
from tools.kaggle_eb01_v7_single_shot import check_configuration
from studio.eb01_v7_storyboard import sha
from studio.comfy_neural import _probe
from studio.eb01_motion_artifact_gate import inspect_clip
from studio.eb01_v7_native_micro_master import collect_and_finish

BRANCH="fix/eb01-s02-private-raw-preservation-20261113"
DATASET="wolf19loner92/eb01-v7-private-production-sources"
PREVIOUS={"original":"wolf19loner92/eb01-v7-native-full-c188f4380731",
 "S03":"wolf19loner92/eb01-v7-s03-only-1e7b0ac41e3a",
 "S04":"wolf19loner92/eb01-v7-s04-only-2333bba16751",
 "S05":"wolf19loner92/eb01-v7-s05-only-75dbb6f053bf",
 "S06_original":"wolf19loner92/eb01-v7-s06-only-beca33b6874e",
 "S06_tail":"wolf19loner92/eb01-v7-s06-tail29-17f5716049d7"}
EXPECTED_S02=("S02_A41","S02_B41","S02_C45")

def call(args,timeout=300):
    cmd=[str(x) for x in args]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    if p.returncode:
        raise RuntimeError("Program "+cmd[0]+" error: "+p.stdout[-500:]+p.stderr[-700:])
    return p.stdout+p.stderr

def save(file:Path,info:dict):
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def status(kernel:str)->str:
    answer=call(["kaggle","kernels","status",kernel],timeout=90)
    found=re.findall(r"KernelWorkerStatus\.([A-Z_]+)",answer)
    if len(found)!=1:raise RuntimeError("Kaggle status ambiguous: "+answer[-250:])
    return found[0]

def await_real(kernel:str,scene:str,receipt:Path,minutes:int=70):
    stop=time.monotonic()+minutes*60
    while time.monotonic()<stop:
        worker=status(kernel)
        print("EB01_ACTUAL_PRIVATE_GPU",scene,kernel,worker,flush=True)
        save(receipt,{"scene":scene,"kernel":kernel,"state":worker,
                      "social_published":False,"complete_film":False})
        if worker=="COMPLETE":return
        if worker in ("ERROR","CANCEL_ACKNOWLEDGED","FAILED","CANCELLED"):
            raise RuntimeError("GPU kernel failed real quality control "+scene+
                               " "+kernel+" "+worker)
        time.sleep(90)
    raise RuntimeError("GPU kernel bounded timeout "+kernel)

def output(kernel:str,dest:Path):
    dest.mkdir(parents=True,exist_ok=False)
    call(["kaggle","kernels","output",kernel,"-p",dest,"--force"],timeout=240)
    return dest

def approved_micro(path:Path,segment:str)->dict:
    spec=SEGMENT[segment]
    sid=spec["shot"]
    movie=path/(sid+".mp4")
    models=path/(sid+".receipt.json")
    job=path/"EB01_V7_SINGLE_SHOT_JOB.json"
    micro=path/"EB01_V7_MICRO_SHOT_RECEIPT.json"
    for p in (movie,models,job,micro):
        if not p.is_file():raise RuntimeError("Genuine micro source/receipt absent "+segment)
    j=json.loads(job.read_text())
    m=json.loads(micro.read_text())
    original=json.loads(models.read_text())
    digest=sha(movie)
    if not (j.get("status")=="ONE_REAL_WAN_SHOT_TECHNICALLY_VERIFIED"
            and j.get("success") is True
            and j.get("shot")==sid
            and j.get("frame_count")==spec["frames"]
            and j.get("clip_sha256")==digest
            and j.get("model")=="wan22-5b"
            and j.get("neural_inference_performed") is True
            and m.get("segment")==segment
            and m.get("source_clip_sha256")==digest
            and original.get("engine")=="neural_comfy"
            and original.get("clip_sha256")==digest
            and original.get("inference_performed") is True
            and original.get("published") is False):
        raise RuntimeError("Cannot verify independent Wan inference "+segment)
    meta=_probe(movie)
    quality=inspect_clip(movie)
    if not (meta.get("frames")==spec["frames"] and meta.get("fps")=="24"
            and quality.get("status")=="NO_NEON_ARTIFACT_FOUND"):
        raise RuntimeError("Model corrupted scene "+segment)
    call(["ffmpeg","-v","error","-xerror","-i",movie,"-f","null","-"],timeout=180)
    return {"segment":segment,"SHA256":digest,"frames":spec["frames"],
            "real_native_wan22_inference":True,"technical_quality":"PASS",
            "human_artistic_approved":False}

def issue(segment:str,root:Path,rev:str,run_id:str)->str:
    spec=SEGMENT[segment]
    if segment not in EXPECTED_S02:raise RuntimeError("Scene not in hard-coded recovery plan")
    nonce="eb01-v7-S02-native-micro-"+run_id+"-"+segment
    # Kaggle normalizes underscores in kernel slugs; always use canonical hyphens.
    # Otherwise push succeeds but kernels.get immediately fails with a 403.
    slug="eb01-v7-"+segment.lower().replace("_","-")+"-"+hashlib.sha256(nonce.encode()).hexdigest()[:12]
    if not re.fullmatch(r"[a-z0-9-]{1,80}",slug):
        raise RuntimeError("Private Kaggle kernel slug is not canonical")
    kernel="wolf19loner92/"+slug
    cfg={"schema":7,"nonce":nonce,"shot":spec["shot"],"segment":segment,
         "model":"wan22-5b","private":True,"new_gpu_jobs":1,
         "source_dataset":DATASET,"revision":rev,"branch":BRANCH,
         "allow_social_publish":False,"allow_regenerate_good_shots":False,
         "quality_retake":True,"seed_variant":spec["seed_variant"]}
    assert check_configuration(cfg)==spec["shot"]
    source=Path("tools/kaggle_eb01_v7_micro_shot.py").read_text(encoding="utf-8")
    needle="__EB01_V7_MICRO_SHOT_CONFIG_B64__"
    if source.count(needle)!=1:raise RuntimeError("Unsafe micro worker template")
    source=source.replace(needle,base64.b64encode(json.dumps(cfg,sort_keys=True).encode()).decode())
    folder=root/("launch_"+segment)
    folder.mkdir()
    (folder/"kernel_main.py").write_text(source,encoding="utf-8")
    meta={"id":kernel,"title":slug,"code_file":"kernel_main.py",
          "language":"python","kernel_type":"script","is_private":"true",
          "enable_gpu":"true","enable_internet":"true",
          "machine_shape":"NvidiaTeslaT4","dataset_sources":[DATASET],
          "competition_sources":[],"kernel_sources":[],"model_sources":[]}
    save(folder/"kernel-metadata.json",meta)
    call(["kaggle","kernels","push","-p",folder,"--timeout","3600",
          "--accelerator","NvidiaTeslaT4"],timeout=380)
    print("ACTUAL_GENUINE_S02_MODEL_JOB_SUBMITTED",segment,kernel,flush=True)
    return kernel

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--revision",required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}",args.revision):
        raise RuntimeError("Exact code Git SHA must be pinned")
    if not re.fullmatch(r"[0-9]+",args.run_id):
        raise RuntimeError("Exact GitHub run id required")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("Kaggle user token not connected")
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    report={"schema":7,"status":"STARTED_GENUINE_S02_NEURAL_RESCUE",
            "S06_micro_kernel":PREVIOUS["S06_tail"],"S02_segments":{},
            "full_movie_ready":False,"published":False,
            "automatic_GPU_restarts":0}
    receipt=root/"EB01_REAL_PROGRESS.json"
    save(receipt,report)
    with tempfile.TemporaryDirectory(prefix="eb01-GENUINE-native-") as tmp:
        base=Path(tmp)
        try:
            # Do not launch another GPU unless S06 already produced a valid
            # genuine 29-frame source with receipt and all pixels checked.
            await_real(PREVIOUS["S06_tail"],"S06_TAIL29",receipt,minutes=70)
            s06=output(PREVIOUS["S06_tail"],base/"S06_tail")
            report["S06_micro"]=approved_micro(s06,"S06_TAIL29")
            save(receipt,report)
            segments={}
            for segment in EXPECTED_S02:
                kernel=issue(segment,base,args.revision,args.run_id)
                report["S02_segments"][segment]={"kernel":kernel,
                                                  "status":"IN_PROGRESS"}
                save(receipt,report)
                await_real(kernel,segment,receipt,minutes=69)
                folder=output(kernel,base/segment)
                report["S02_segments"][segment]={
                    "kernel":kernel,**approved_micro(folder,segment)}
                segments[segment]=folder
                save(receipt,report)
            # Only after all genuine source clips are proven, obtain
            # original SHA-locked portraits/six WAVs and compose 7 scenes.
            sources={}
            for label,kernel in PREVIOUS.items():
                if label=="S06_tail":continue
                sources[label]=output(kernel,base/label)
            dataset=base/"original_dataset"
            dataset.mkdir()
            call(["kaggle","datasets","download","-d",DATASET,
                  "-p",dataset,"--unzip"],timeout=480)
            mapping={"S00":sources["original"],"S01":sources["original"],
                     "S03":sources["S03"],"S04":sources["S04"],
                     "S05":sources["S05"]}
            report["finished_film"]=collect_and_finish(
                dataset,mapping,sources["S06_original"],s06,segments,
                root/"complete_movie")
            report["status"]="FULL_TRUE_NEURAL_7_SCENE_MP4_OWNER_REVIEW_READY"
            report["full_movie_ready"]=True
            save(receipt,report)
            print("EB01_COMPLETED_GENUINE_ELIZABETH_BONES_MASTER",
                  report["finished_film"]["video_sha256"],flush=True)
        except BaseException as exc:
            report["status"]="BLOCKED_BUT_PRIVATE_NEURAL_SOURCES_PRESERVED"
            report["error"]=type(exc).__name__+": "+str(exc)[:800]
            save(receipt,report)
            raise

if __name__=="__main__":
    main()
