"""One-shot Kaggle CPU preflight controller for public Tencent model source access.
No GPU or user game originals used. Outputs safe booleans and HTTP enum only.
"""
from pathlib import Path
import argparse,json,shutil,subprocess,time,re,os
import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi

SLUG="tflex-v98-cpu-public-model-probe"
TITLE="TFlex V98 CPU Hunyuan Import Probe"
FOLDER=Path("dist/tflex_v98_cpu_preflight")

def owner():
    name=kagglehub.auth.whoami(verbose=False).get("username")
    if not name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,29}",name):
        raise RuntimeError("Authenticated Kaggle identity not found")
    return name

def create():
    u=owner();FOLDER.mkdir(parents=True,exist_ok=True)
    shutil.copyfile("ops/tflex_v98_kaggle_cpu_hunyuan_import.py",FOLDER/"main.py")
    meta={"id":u+"/"+SLUG,"title":TITLE,"code_file":"main.py",
          "language":"python","kernel_type":"script","is_private":"true",
          "enable_gpu":"false","enable_internet":"true","machine_shape":"",
          "dataset_sources":[],"competition_sources":[],"kernel_sources":[],"model_sources":[]}
    (FOLDER/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
    print("TFLEX_V98_CPU_PRIVATE_METADATA_READY")

def launch():
    create()
    api=KaggleApi();api.authenticate()
    result=api.kernels_push(str(FOLDER),timeout="600",no_run=False)
    if not result or getattr(result,"error",None):
        raise RuntimeError("Cannot start CPU preflight")
    print("TFLEX_V98_CPU_PREFLIGHT_SUBMITTED")

def wait():
    u=owner(); api=KaggleApi();api.authenticate()
    stop=time.monotonic()+630
    while time.monotonic()<stop:
        response=api.kernels_status(u+"/"+SLUG)
        state=str(getattr(response,"status","")).upper()
        if "COMPLETE" in state:print("TFLEX_V98_CPU_PREFLIGHT_REMOTE_COMPLETE");return
        if "ERROR" in state:raise RuntimeError("CPU preflight errored")
        time.sleep(30)
    raise TimeoutError("Kaggle CPU preflight did not complete")

def verify():
    import subprocess
    u=owner();out=Path("dist/tflex_v98_cpu_results");out.mkdir(parents=True,exist_ok=True)
    process=subprocess.run(["kaggle","kernels","output",u+"/"+SLUG,
                            "-p",str(out),"--file-pattern",r"^tflex_v98_cpu_official_hunyuan_import[.]json$"],
                            text=True,capture_output=True,timeout=120)
    if process.returncode:raise RuntimeError("Kaggle CPU report download failed")
    matches=list(out.rglob("tflex_v98_cpu_official_hunyuan_import.json"))
    if len(matches)!=1:raise FileNotFoundError("Private CPU preflight result absent")
    data=json.loads(matches[0].read_text())
    assert data.get("user_screenshots_uploaded") is False
    print("TFLEX_V98_CPU_MODEL_NETWORK_RESULTS",json.dumps(data,sort_keys=True))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("launch","wait","verify"))
    {"launch":launch,"wait":wait,"verify":verify}[p.parse_args().action]()
