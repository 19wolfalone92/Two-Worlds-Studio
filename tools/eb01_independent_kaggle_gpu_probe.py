"""Submit ONE private, no-media GPU hardware micro-probe per legitimate Kaggle user.

Authorized for EB01 source-independent T4 hardware measurements only.
Does not load neural models, copy source, submit another worker, or publish.
Each account has its own secret and its own immutable job slug.
CLI output is never echoed (it can contain sensitive account info).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from tools.eb01_kaggle2_auth_probe import query

ACCOUNT_IDS = ("owner", "collaborator")
STATUS_EXPR = re.compile(r"KernelWorkerStatus\.(COMPLETE|QUEUED|RUNNING|ERROR|CANCELLED|CANCEL_ACKNOWLEDGED)")
CODE = r'''import json
from pathlib import Path
report={"schema":1,"cuda_available":False,"gpu_device_count":0,
        "per_gpu_gib":[],"aggregated_vram_not_one_gpu":True,
        "neural_model_loaded":False,"private_actor_media_loaded":False,
        "movie_ready":False}
try:
    import torch
    if torch.cuda.is_available():
        report["cuda_available"]=True
        report["gpu_device_count"]=torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            p=torch.cuda.get_device_properties(i)
            free,total=torch.cuda.mem_get_info(i)
            report["per_gpu_gib"].append({
                "index":i, "gpu_family":"T4" if "T4" in p.name.upper() else "OTHER",
                "physical_gib":round(p.total_memory/(1024**3),3),
                "free_gib":round(free/(1024**3),3)})
    report["status"]="REAL_GPU_MEASURED" if report["cuda_available"] else "NO_CUDA"
except Exception as exc:
    report["status"]="GPU_PROBE_FAILED_"+type(exc).__name__
Path("/kaggle/working/EB01_HARDWARE_ONLY_SAFE.json").write_text(json.dumps(report,sort_keys=True))
print("EB01_T4_HARDWARE_MEASUREMENT_ONLY",json.dumps(report,sort_keys=True))
'''


class GPUProbeBlocked(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def safe_call(args, token, timeout=180):
    if not token:
        raise GPUProbeBlocked("KAGGLE_TOKEN_MISSING")
    env=os.environ.copy()
    env["KAGGLE_API_TOKEN"]=token
    env.pop("KAGGLE_USERNAME",None)
    try:
        p=subprocess.run(args,env=env,stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         text=True,timeout=timeout,check=False)
    except subprocess.TimeoutExpired:
        raise GPUProbeBlocked("KAGGLE_CLI_TIMEOUT") from None
    except OSError:
        raise GPUProbeBlocked("KAGGLE_CLI_UNAVAILABLE") from None
    if p.returncode:
        raw=((p.stdout or "")+" "+(p.stderr or "")).lower()
        if "429" in raw:
            raise GPUProbeBlocked("KAGGLE_RATE_LIMITED")
        if "403" in raw or "forbidden" in raw:
            raise GPUProbeBlocked("KAGGLE_FORBIDDEN")
        if "401" in raw:
            raise GPUProbeBlocked("KAGGLE_AUTH_DENIED")
        if "quota" in raw or "gpu limit" in raw:
            raise GPUProbeBlocked("KAGGLE_GPU_QUOTA_EXHAUSTED")
        raise GPUProbeBlocked("KAGGLE_COMMAND_FAILED")
    return p.stdout or ""


def probe_alias(alias, token, run_id, scratch,
                poll_seconds=270):
    result={"schema":1,"account_alias":alias,"repo_source_media_copied":False,
            "gpu_submission_attempts":0,"gpu_movie_generation":False,
            "published":False,"gpu_measured":False}
    try:
        if alias not in ACCOUNT_IDS:
            raise GPUProbeBlocked("UNKNOWN_ACCOUNT_ALIAS")
        if not re.fullmatch(r"[1-9][0-9]{5,20}",run_id):
            raise GPUProbeBlocked("INVALID_GITHUB_RUN_ID")
        state,handle=query(token)
        if state!="AUTHENTICATED_READ_ONLY" or not handle:
            raise GPUProbeBlocked("AUTHENTICATED_KAGGLE_ACCOUNT_NOT_FOUND")
        nonce=hashlib.sha256(("eb01-hw-"+run_id+"-"+alias).encode()).hexdigest()[:12]
        slug="eb01-hw-probe-"+nonce
        dest=scratch/"kernel"
        dest.mkdir(mode=0o700,parents=True,exist_ok=False)
        (dest/"kernel_main.py").write_text(CODE,encoding="utf-8")
        meta={
            "id":handle+"/"+slug,
            "title":"EB01 Private GPU Hardware Probe "+nonce,
            "code_file":"kernel_main.py",
            "language":"python", "kernel_type":"script",
            "is_private":"true", "enable_gpu":"true",
            "enable_internet":"false", "machine_shape":"NvidiaTeslaT4",
            "dataset_sources":[], "competition_sources":[],
            "kernel_sources":[], "model_sources":[]
        }
        (dest/"kernel-metadata.json").write_text(json.dumps(meta),encoding="utf-8")
        # Exactly one new short, PRIVATE Kaggle kernel for THIS account.
        result["gpu_submission_attempts"]=1
        safe_call(["kaggle","kernels","push","-p",str(dest),
                   "--timeout","3600","--accelerator","NvidiaTeslaT4"],
                  token,timeout=200)
        result["status"]="PRIVATE_GPU_PROBE_SUBMITTED_NOT_VERIFIED"
        deadline=time.monotonic()+poll_seconds
        while time.monotonic()<deadline:
            out=safe_call(["kaggle","kernels","status",meta["id"]],
                          token,timeout=65)
            found=STATUS_EXPR.findall(out)
            if len(found)!=1:
                raise GPUProbeBlocked("UNKNOWN_KAGGLE_KERNEL_STATUS")
            current=found[0]
            result["kaggle_state"]=current
            if current=="COMPLETE":
                output=scratch/"output"
                output.mkdir(mode=0o700,parents=True,exist_ok=False)
                safe_call(["kaggle","kernels","output",meta["id"],
                           "-p",str(output),"--force"],token,timeout=200)
                audit=output/"EB01_HARDWARE_ONLY_SAFE.json"
                if not audit.is_file():
                    raise GPUProbeBlocked("GPU_REPORT_NOT_AVAILABLE")
                data=json.loads(audit.read_text())
                if (data.get("status")!="REAL_GPU_MEASURED" or
                    data.get("cuda_available") is not True or
                    type(data.get("gpu_device_count")) is not int or
                    not 1<=data["gpu_device_count"]<=2 or
                    not isinstance(data.get("per_gpu_gib"),list) or
                    len(data["per_gpu_gib"])!=data["gpu_device_count"]):
                    raise GPUProbeBlocked("NO_VERIFIABLE_T4_HARDWARE")
                result["gpu_measured"]=True
                result["per_gpu_gib"]=data["per_gpu_gib"]
                result["gpu_device_count"]=data["gpu_device_count"]
                result["status"]="REAL_PRIVATE_GPU_HARDWARE_VERIFIED_NO_VIDEO"
                return result
            if current in ("ERROR","CANCELLED","CANCEL_ACKNOWLEDGED"):
                raise GPUProbeBlocked("KAGGLE_GPU_PROBE_FAILED")
            time.sleep(18)
        result["status"]="GPU_JOB_QUEUED_OR_RUNNING_NO_HARDWARE_PROOF_YET"
    except (GPUProbeBlocked,ValueError,KeyError,json.JSONDecodeError) as exc:
        result["status"]="BLOCKED_"+(
            exc.reason if isinstance(exc,GPUProbeBlocked) else "INVALID_PRIVATE_REPORT")
    return result


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--account-alias",choices=ACCOUNT_IDS,required=True)
    args=parser.parse_args()
    if not os.environ.get("RUNNER_TEMP"):
        raise RuntimeError("Ephemeral GitHub runner only")
    os.umask(0o077)
    root=Path(os.environ["RUNNER_TEMP"]).resolve()
    workspace=Path(os.environ.get("GITHUB_WORKSPACE","/__none__")).resolve()
    if workspace==root or workspace in root.parents or root in workspace.parents:
        raise RuntimeError("Private Kaggle kernel must not stage in public checkout")
    with tempfile.TemporaryDirectory(dir=root,prefix="eb01-hw-") as tmp:
        result=probe_alias(args.account_alias,
                           os.environ.get("EB01_KAGGLE_ACCOUNT_TOKEN",""),
                           os.environ.get("GITHUB_RUN_ID",""),
                           Path(tmp))
    (root/("EB01_HW_"+args.account_alias+".json")).write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("EB01_INDEPENDENT_KAGGLE_GPU_HARDWARE",json.dumps(result,sort_keys=True))
    return 0 if result["status"] in (
        "REAL_PRIVATE_GPU_HARDWARE_VERIFIED_NO_VIDEO",
        "GPU_JOB_QUEUED_OR_RUNNING_NO_HARDWARE_PROOF_YET") else 2


if __name__=="__main__":
    raise SystemExit(main())
