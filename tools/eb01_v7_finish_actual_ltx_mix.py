"""CPU-only final real mixed neural master: Wan2.2 + genuine LTX2B S02.

Wait for exactly one already submitted native LTX Kaggle job. Collect all
seven independently source-verified moving clips and original six Qwen WAVs;
repair S06 from 60 clean old Wan frames + verified genuine 29-frame Wan take.
No fresh GPU, no static fallback, no social publication, no fake success.
"""
from __future__ import annotations
import argparse,json,os,re,shutil,subprocess,tempfile,time
from pathlib import Path

from studio.eb01_v7_storyboard import SHOTS,SRC_SHA,VOICE_SHA,gather_private,sha
from studio.comfy_neural import _probe
from studio.ltx_take import import_take as verify_ltx
from studio.eb01_v7_cpu_final_recovery import OLD_GENUINE_SHA
from studio.eb01_v7_native_micro_master import (
 genuine_source,trimmed_segments,S06_ORIGINAL_VARIANT1_SHA,S03_GENUINE_SHA,S05_GENUINE_SHA,
)
from tools import kaggle_eb01_v7_full_entry as composer

LTX="wolf19loner92/eb01-s02-ltx2b-real-103857643f19"
DATASET="wolf19loner92/eb01-v7-private-production-sources"
SOURCES={
 "ORIGINAL":"wolf19loner92/eb01-v7-native-full-c188f4380731",
 "S03":"wolf19loner92/eb01-v7-s03-only-1e7b0ac41e3a",
 "S04":"wolf19loner92/eb01-v7-s04-only-2333bba16751",
 "S05":"wolf19loner92/eb01-v7-s05-only-75dbb6f053bf",
 "S06_RAW":"wolf19loner92/eb01-v7-s06-only-beca33b6874e",
 "S06_TAIL":"wolf19loner92/eb01-v7-s06-tail29-17f5716049d7",
}
def run(args,timeout=240):
    cmd=[str(a) for a in args]
    p=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(cmd[0]+" error "+p.stdout[-400:]+p.stderr[-900:])
    return p.stdout+p.stderr

def output(kernel:str,directory:Path):
    directory.mkdir(parents=True,exist_ok=False)
    run(["kaggle","kernels","output",kernel,"-p",directory,"--force"],timeout=360)
    return directory

def one(folder:Path,filename:str)->Path:
    matches=list(folder.rglob(filename))
    if len(matches)!=1:raise RuntimeError("Missing/duplicate genuine source "+filename)
    return matches[0]

def status(kernel:str)->str:
    s=run(["kaggle","kernels","status",kernel],timeout=80)
    x=re.findall(r"KernelWorkerStatus\.([A-Z_]+)",s)
    if len(x)!=1:raise RuntimeError("Ambiguous genuine GPU status "+s[:250])
    return x[0]

def await_original(output_json:Path):
    start=time.monotonic()
    while time.monotonic()-start<72*60:
        stage=status(LTX)
        print("EB01_ACTUAL_LTX_125_GPU_QUALITY_RESULT",LTX,stage,flush=True)
        output_json.write_text(json.dumps({"full_movie_ready":False,
          "source_kernel":LTX,"state":stage,"published":False},indent=2))
        if stage=="COMPLETE":return
        if stage in ("ERROR","FAILED","CANCELLED","CANCEL_ACKNOWLEDGED"):
            raise RuntimeError("Authentic private LTX 125-frame inference failed "+stage)
        time.sleep(90)
    raise RuntimeError("LTX private GPU response deadline exceeded")

def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--output",type=Path,required=True)
    a=arg.parse_args()
    dest=a.output
    if dest.exists():raise RuntimeError("Never overwrite a previous film")
    if not os.environ.get("KAGGLE_API_TOKEN"):raise RuntimeError("Private Kaggle token absent")
    dest.mkdir(parents=True)
    report={"schema":7,"status":"WAITING_FOR_EXISTING_REAL_S02_GPU", "published":False,
            "full_movie_ready":False,"no_new_gpu_jobs":True,"ltx_kernel":LTX}
    audit=dest/"EB01_REAL_FULL_NEURAL_MOVIE_STATUS.json"
    audit.write_text(json.dumps(report,indent=2))
    with tempfile.TemporaryDirectory(prefix="EB01_ACTUAL_FILM_") as tmp:
      root=Path(tmp)
      try:
        await_original(audit)
        raw={}
        for name,kernel in [("LTX",LTX),*SOURCES.items()]:
            raw[name]=output(kernel,root/name)
        dataset=root/"private_dataset"
        dataset.mkdir()
        run(["kaggle","datasets","download","-d",DATASET,
             "-p",dataset,"--unzip"],timeout=500)
        studio=root/"original_private_studio"
        lock=gather_private(dataset,studio)
        if lock["source_sha256"]!=SRC_SHA or lock["voice_sha256"]!=VOICE_SHA:
            raise RuntimeError("Canonical photo/6 Russian voice SOURCE SHA changed")
        clips=studio/"clips"
        clips.mkdir()
        proof_by_scene={}
        for sid,known in [("S00",OLD_GENUINE_SHA["S00"]),
                          ("S01",OLD_GENUINE_SHA["S01"]),
                          ("S03",S03_GENUINE_SHA),("S04",None),
                          ("S05",S05_GENUINE_SHA)]:
            source=raw["ORIGINAL"] if sid in ("S00","S01") else raw[sid]
            pic=one(source,sid+".mp4")
            receipt=one(source,sid+".receipt.json")
            metadata=genuine_source(sid,pic,receipt,
                 next(x[2] for x in SHOTS if x[0]==sid),
                 lock["scenes"][sid]["reference_sha256"],known)
            shutil.copy2(pic,clips/(sid+".mp4"))
            shutil.copy2(receipt,clips/(sid+".receipt.json"))
            proof_by_scene[sid]={"clip_sha256":metadata["original_sha256"],
                                 "model":"Wan2.2-5B original signed source"}
            print("EB01_VERIFIED_EXISTING_REAL_NATIVE_SCENE",sid,
                  metadata["original_sha256"],flush=True)
        # Assemble exactly 89 full native S06 actor frames from two REAL
        # independent Wan sources; no repeats, inpainting or still frames.
        raw_source=one(raw["S06_RAW"],"S06.original_wan_raw.mp4")
        raw_receipt=one(raw["S06_RAW"],"S06.original_wan_raw.receipt.json")
        tail=one(raw["S06_TAIL"],"S06.mp4")
        tail_receipt=one(raw["S06_TAIL"],"S06.receipt.json")
        used=[
         {"name":"S06_V1_PREFIX","source":raw_source,"receipt":raw_receipt,
          "original_frames":93,"used_frames":60,
          "expected_sha":S06_ORIGINAL_VARIANT1_SHA},
         {"name":"S06_TAIL29","source":tail,"receipt":tail_receipt,
          "original_frames":29,"used_frames":29},
        ]
        restored=trimmed_segments("S06",used,root/"s06_genuine_restored",
                           lock["scenes"]["S06"]["reference_sha256"])
        shutil.copy2(root/"s06_genuine_restored"/"S06.mp4",clips/"S06.mp4")
        shutil.copy2(root/"s06_genuine_restored"/"S06.receipt.json",
                     clips/"S06.receipt.json")
        proof_by_scene["S06"]={"clip_sha256":restored["final_sha256"],
                               "model":"Wan2.2 two genuine native segments"}
        # The alternative S02 is NOT Wan. Verify the official source-engine
        # receipt before passing through Studio's distinct neural_ltx engine.
        s02=one(raw["LTX"],"S02.mp4")
        receipt=one(raw["LTX"],"EB01_S02_LTX_PRIVATE_JOB.json")
        model_proof=verify_ltx(s02,receipt,studio/"frames"/"S02.png",
                               duration=125/24)
        shutil.copy2(s02,clips/"S02.mp4")
        shutil.copy2(receipt,clips/"S02.receipt.json")
        proof_by_scene["S02"]={"clip_sha256":model_proof["clip_sha256"],
                      "model":"Lightricks/LTX-Video-2B-distilled-0.9.6"}
        # Actual 7-scene photorealistic Studio render, native original WAVs
        # and six verified Russian subtitle cues. Always owner review, no publish.
        film=dest/"ELIZABETH_BONES_SOBESEDOVANIE_ACTUAL_ANIMATED_7SCENE_RU.mp4"
        composer.ROOT=studio
        composer.FINAL=film
        composer.FILM_QA=dest/"EB01_REAL_HYBRID_NEURAL_QA.json"
        master=composer.assemble_native({"private":True},lock,proof_by_scene)
        if (not film.is_file() or film.stat().st_size<500000
             or master.get("real_I2V_shots")!=7
             or master.get("russian_voice_lines")!=6
             or master.get("subtitles")!=6
             or master.get("source_engine_by_scene",{}).get("S02")!="neural_ltx"
             or master.get("full_film_sha256")!=sha(film)):
            raise RuntimeError("Studio final real 7-scene original Russian film gate failed")
        report.update({"status":"ACTUAL_FULL_ANIMATED_7SCENE_RUSSIAN_MP4_OWNER_REVIEW",
           "full_movie_ready":True,"real_neural_scenes":7,
           "original_qwen_ru_voice_lines":6,"subtitle_cues":6,
           "movie_name":film.name,"movie_sha256":sha(film),
           "sources":proof_by_scene,"artistically_approved":False,"published":False})
        audit.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        print("EB01_ACTUAL_FULL_7SCENE_REAL_GPU_ANIMATED_RUSSIAN_MP4",
              report["movie_sha256"],flush=True)
      except BaseException as failure:
        report["status"]="BLOCKED_NO_FALSE_COMPLETE_FILM"
        report["failure"]=type(failure).__name__+": "+str(failure)[:1000]
        audit.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        raise

if __name__=="__main__":
    main()
