"""PRIVATE one-shot Kaggle T4 GPU capacity proof for the T-Flex 200mm figure.

Only executes code that checks CUDA and writes a non-sensitive hardware report.
Original user pictures, model weights, printed secrets or personal account
handles are NEVER placed in a public GitHub artifact, command log or repository.
Uses separately authorized KGAT on each legitimate Kaggle account.
"""
from __future__ import annotations
import argparse,contextlib,io,json,os,re,subprocess,sys,time
from pathlib import Path

KERNEL_SLUG="tflex-v9-200mm-gpu-hardware-check"

def owner():
    import kagglehub
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        u=kagglehub.auth.whoami(verbose=False).get("username")
    if not isinstance(u,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,29}",u):
        raise RuntimeError("authenticated Kaggle account identity unavailable")
    return u

def sanitized_error(output):
    text=output.lower()
    if "429" in text or "rate limit" in text or "too many requests" in text:return "RATE_LIMIT"
    if "401" in text or "unauthorized" in text:return "AUTH_401"
    if "403" in text or "forbidden" in text:return "ACL_403"
    if "quota" in text or "out of gpu" in text:return "QUOTA_OR_GPU_UNAVAILABLE"
    if "timeout" in text:return "TIMEOUT"
    if "permission" in text:return "PERMISSION"
    if "not found" in text or "404" in text:return "RESOURCE_MISSING"
    return "UNCLASSIFIED_SAFE_ERROR"

def run_safe_command(args,limit=120):
    try:
        r=subprocess.run(args,text=True,capture_output=True,timeout=limit)
    except subprocess.TimeoutExpired:
        raise RuntimeError("KAGGLE_CLI_TIMEOUT") from None
    if r.returncode:
        raise RuntimeError("KAGGLE_"+sanitized_error((r.stderr or "")+"\\n"+(r.stdout or "")))
    return r.stdout

def prepare():
    name=owner()
    root=Path("dist/tflex_v9_gpu_probe")
    root.mkdir(parents=True,exist_ok=True)
    script='''"""Only hardware test for one PRIVATE Kaggle account."""
import json,torch,time
from pathlib import Path
devices=[]
for i in range(torch.cuda.device_count()):
    props=torch.cuda.get_device_properties(i)
    devices.append({"ordinal":i,"gpu_family":props.name,
      "vram_gib":round(props.total_memory/(1024**3),2)})
data={"gpu_count":len(devices),"devices":devices,
      "cuda":str(torch.version.cuda),"torch":str(torch.__version__),
      "image_references_received":False,"real_3d_model_generated":False}
p=Path("/kaggle/working/tflex_v9_hardware_proof.json")
p.write_text(json.dumps(data,indent=2))
print("PRIVATE_TFLEX_GPU_PROOF_COUNT",len(devices))
if len(devices)==0:raise RuntimeError("No assigned accelerator")
'''
    (root/"probe.py").write_text(script,encoding="utf8")
    info={"id":name+"/"+KERNEL_SLUG,
      "title":"TFlex 200mm private T4 hardware check",
      "code_file":"probe.py","language":"python","kernel_type":"script",
      "is_private":"true","enable_gpu":"true","enable_internet":"false",
      "machine_shape":"NvidiaTeslaT4","dataset_sources":[],
      "competition_sources":[],"kernel_sources":[],"model_sources":[]}
    (root/"kernel-metadata.json").write_text(json.dumps(info,indent=2))
    print("PRIVATE_KAGGLE_HARDWARE_JOB_PREPARED true images=NONE")
    return info["id"]

def launch():
    prepare()
    # Kaggle CLI sometimes prints a "Kernel push error" and exits code zero.
    # Use the official Python SDK response instead; validate response.error.
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi()
        api.authenticate()
        result=api.kernels_push("dist/tflex_v9_gpu_probe",
                                timeout="1200",acc="NvidiaTeslaT4",no_run=False)
    if result is None:
        raise RuntimeError("KAGGLE_KERNEL_PUSH_RETURNED_NONE")
    problem=getattr(result,"error",None)
    if problem:
        raise RuntimeError("KAGGLE_KERNEL_PUSH_REJECTED_"+sanitized_error(str(problem)))
    print("PRIVATE_KAGGLE_GPU_JOB_SUBMITTED true")
    return result

def poll(max_seconds=1000):
    key=owner()+"/"+KERNEL_SLUG
    end=time.monotonic()+max_seconds
    prev=None
    while time.monotonic()<end:
        result=run_safe_command(["kaggle","kernels","status",key],limit=60)
        match=re.search(r"KERNELWORKERSTATUS[.]([A-Z_]+)",result.upper())
        status=match.group(1) if match else "UNRECOGNIZED"
        if status!=prev:
            print("PRIVATE_KAGGLE_GPU_JOB_STATUS",status)
            prev=status
        if status=="COMPLETE":
            print("PRIVATE_KAGGLE_GPU_HARDWARE_RUN_COMPLETED true")
            return
        if status in ("ERROR","CANCEL_REQUESTED","CANCEL_ACKNOWLEDGED"):
            raise RuntimeError("Private Kaggle job status "+status)
        if status not in ("QUEUED","RUNNING","NEW_SCRIPT"):
            raise RuntimeError("Unrecognized Kaggle GPU status")
        time.sleep(55)
    raise TimeoutError("Private Kaggle GPU check did not complete within deadline")

def verify():
    key=owner()+"/"+KERNEL_SLUG
    target=Path("dist/tflex_v9_hardware_download")
    target.mkdir(parents=True,exist_ok=True)
    run_safe_command(["kaggle","kernels","output",key,"-p",str(target),
       "--file-pattern",r"^tflex_v9_hardware_proof[.]json$"],limit=120)
    matches=list(target.rglob("tflex_v9_hardware_proof.json"))
    if len(matches)!=1:raise RuntimeError("Kaggle hardware report missing")
    report=json.loads(matches[0].read_text())
    count=report.get("gpu_count")
    if not isinstance(count,int) or count<1 or count>4:raise RuntimeError("No valid T4 GPU measurement")
    if report.get("image_references_received") is not False:
        raise RuntimeError("Private art should not be sent by hardware check")
    for x in report.get("devices",[]):
        if not isinstance(x.get("vram_gib"),(int,float)):
            raise RuntimeError("Invalid VRAM result")
    result={"gpu_devices_observed":count,"vram_gib_each":[x["vram_gib"] for x in report["devices"]],
            "private_hardware_report_verified":True,"original_images_uploaded":False,
            "3d_character_generated":False}
    print("TFLEX_SAFE_KAGGLE_HARDWARE_RESULT",json.dumps(result,sort_keys=True))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("prepare","launch","poll","verify"))
    args=p.parse_args()
    if args.mode=="prepare":prepare()
    elif args.mode=="launch":launch()
    elif args.mode=="poll":poll()
    else:verify()
