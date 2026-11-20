"""Run ONE Kaggle T4 Hunyuan3D multiview benchmark on PUBLIC example images.
Never contains original user reference images; private Kaggle notebooks only.
"""
from __future__ import annotations
import argparse,hashlib,io,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
from ops.tflex_v9_kaggle_gpu_probe import owner,sanitized_error,run_safe_command

SLUG="tflex-v94-hunyuan-official-public-sample"
TITLE="TFlex V94 Hunyuan Official Public Sample"
OUT=Path("dist/tflex_v94_hunyuan_demo")

def prepare():
    who=owner()
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copyfile("ops/tflex_v94_public_sample_hunyuan_kernel.py",OUT/"pilot.py")
    meta={"id":who+"/"+SLUG,"title":TITLE,"code_file":"pilot.py",
          "language":"python","kernel_type":"script",
          "is_private":"true","enable_gpu":"true","enable_internet":"true",
          "machine_shape":"NvidiaTeslaT4","dataset_sources":[],
          "competition_sources":[],"kernel_sources":[],"model_sources":[]}
    (OUT/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
    print("TFLEX_OFFICIAL_PUBLIC_DEMO_PREPARED true private_source_images_sent=false")

def launch():
    prepare()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    response=api.kernels_push(str(OUT),timeout="4500",acc="NvidiaTeslaT4")
    if response is None:raise RuntimeError("Kaggle SaveKernel returned no response")
    if getattr(response,"error",None):
        raise RuntimeError("KAGGLE_REJECTED_"+sanitized_error(str(response.error)))
    print("TFLEX_V94_OFFICIAL_PUBLIC_DEMO_SUBMITTED true")

def status():
    key=owner()+"/"+SLUG
    previous=""
    end=time.monotonic()+1900
    while time.monotonic()<end:
        raw=run_safe_command(["kaggle","kernels","status",key],limit=90)
        m=re.search(r"KERNELWORKERSTATUS[.]([A-Z_]+)",raw.upper())
        state=m.group(1) if m else "UNRECOGNIZED"
        if state!=previous:
            print("TFLEX_V94_DEMO_STATE",state,flush=True)
            previous=state
        if state=="COMPLETE":return
        if state in ("ERROR","CANCEL_REQUESTED","CANCEL_ACKNOWLEDGED"):
            raise RuntimeError("Kaggle demo job failed in state "+state)
        if state not in ("RUNNING","QUEUED","NEW_SCRIPT"):raise RuntimeError("Unknown Kaggle state")
        time.sleep(80)
    raise TimeoutError("Kaggle sample pilot still in progress; no false success")

def verify():
    who=owner()
    d=Path("dist/tflex_v94_result");d.mkdir(parents=True,exist_ok=True)
    run_safe_command(["kaggle","kernels","output",who+"/"+SLUG,"-p",str(d),
      "--file-pattern",r"^tflex_v94_public_sample_technical_pilot[.]json$"],limit=200)
    src=list(d.rglob("tflex_v94_public_sample_technical_pilot.json"))
    if len(src)!=1:raise FileNotFoundError("Private Kaggle GPU report missing")
    r=json.loads(src[0].read_text())
    safe={k:r.get(k) for k in ("gpu_count","model_loaded","inference_executed",
        "glb_generated","faces","gpu_peak_allocated_gib","seconds","error_type","error_phase")}
    print("TFLEX_V94_REAL_GPU_SHAPE_PILOT",json.dumps(safe,sort_keys=True))
    if not r.get("glb_generated") or r.get("faces",0)<3000:
        raise RuntimeError("Hunyuan official sample pilot failed to generate mesh")
    if r.get("real_personal_figure_generated") is not False:
        raise RuntimeError("This sample pilot cannot claim user's figure")
    return safe

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","launch","status","verify"))
    mode=parser.parse_args().mode
    {"prepare":prepare,"launch":launch,"status":status,"verify":verify}[mode]()
