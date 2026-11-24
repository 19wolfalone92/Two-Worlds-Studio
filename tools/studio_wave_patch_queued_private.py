"""One-time safety patch to an existing *QUEUED* private Kaggle Studio notebook.

The notebook is pinned to an older Studio revision whose EB01 VAE patch
expects EB01_STAGE_END_MODEL_LOAD. The wave worker uses WAVE_STAGE_...,
so the queued kernel would fail *before producing frames*. Rename the
sentinel only during the old compatible patch and restore it afterward.

Never alter source reference bytes, never create a second kernel, never
touch a RUNNING job, never publish private notebook or user photograph.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path
from tools.studio_wave_status_gate import KERNEL,SOURCE_SHA256,parse_state,JobStateError

OLD='        native_patch_sha=patch_native_sources(source,stages,LTX/"ltx_video/models/autoencoders/vae.py")'
NEW='''        prefix_wave='    print("WAVE_STAGE_END_MODEL_LOAD",flush=True)'
        prefix_eb01='    print("EB01_STAGE_END_MODEL_LOAD",flush=True)'
        original_inference=source.read_text(encoding="utf-8")
        if original_inference.count(prefix_wave)!=1 or prefix_eb01 in original_inference:
            raise RuntimeError("Studio native VAE stage guard changed")
        source.write_text(original_inference.replace(prefix_wave,prefix_eb01),encoding="utf-8")
        native_patch_sha=patch_native_sources(source,stages,LTX/"ltx_video/models/autoencoders/vae.py")
        fixed_inference=source.read_text(encoding="utf-8")
        if fixed_inference.count(prefix_eb01)!=1:
            raise RuntimeError("Native T4 patch did not preserve expected stage")
        source.write_text(fixed_inference.replace(prefix_eb01,prefix_wave),encoding="utf-8")'''

def update_private_script(text:str,*,image_sha:str=SOURCE_SHA256)->str:
    if text.count(OLD)!=1 or text.count('from studio.eb01_native_t4_profile import patch_native_sources')<1:
        raise JobStateError("Not the exact known broken private native T4 script")
    if NEW in text:
        raise JobStateError("Private notebook already patched; do not resubmit")
    found=re.findall(r'^IMAGE_B64="([A-Za-z0-9+/=]+)"$',text,re.MULTILINE)
    if len(found)!=1:
        raise JobStateError("No unique original private actor reference")
    digest=hashlib.sha256(base64.b64decode(found[0],validate=True)).hexdigest()
    if digest!=image_sha:
        raise JobStateError("Original private portrait differs from user-approved source")
    revised=text.replace(OLD,NEW)
    compile(revised,"<private_studio_native_worker>","exec")
    if revised.count("kaggle kernels push") or revised.count("kaggle kernels pull"):
        raise JobStateError("GPU notebook unexpectedly contains self-submitting code")
    return revised

def run_cli(argv:list[str],timeout=200)->str:
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    if p.returncode:
        raise JobStateError("PRIVATE_KAGGLE_CLI_NONZERO")
    return p.stdout

def patch_queued_once(directory:Path)->dict:
    state=parse_state(run_cli(["kaggle","kernels","status",KERNEL]))
    if state!="QUEUED":
        return {"state":state,"patched":False,"submitted":False,
                "reason":"RUNNING_OR_TERMINAL_JOB_NEVER_OVERWRITTEN"}
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    run_cli(["kaggle","kernels","pull",KERNEL,"-p",str(directory),"-m"],timeout=240)
    files=list(directory.glob("*.py"))
    if len(files)!=1:
        raise JobStateError("Ambiguous original notebook pulled from private Kaggle")
    worker=files[0]
    modified=update_private_script(worker.read_text(encoding="utf-8"))
    worker.write_text(modified,encoding="utf-8")
    meta={"id":KERNEL,"title":KERNEL.split("/")[1],
          "code_file":worker.name,"language":"python","kernel_type":"script",
          "is_private":"true","enable_gpu":"true","enable_internet":"true",
          "machine_shape":"NvidiaTeslaT4","dataset_sources":[],
          "competition_sources":[],"kernel_sources":[],"model_sources":[]}
    (directory/"kernel-metadata.json").write_text(json.dumps(meta))
    # Avoid even the tiny race when a T4 is assigned during private code inspection.
    state_just_before_push=parse_state(run_cli(["kaggle","kernels","status",KERNEL]))
    if state_just_before_push!="QUEUED":
        return {"state":state_just_before_push,"patched":True,"submitted":False,
                "reason":"GPU_JOB_BECAME_ACTIVE_DURING_SAFE_PATCH"}
    run_cli(["kaggle","kernels","push","-p",str(directory),
             "--timeout","3600","--accelerator","NvidiaTeslaT4"],timeout=240)
    return {"state":"QUEUED_OR_RESCHEDULED","patched":True,"submitted":True,
            "same_original_kernel":KERNEL,"duplicate_kernel_created":False,
            "private_reference_unchanged":True,"new_public_media_created":False,
            "real_neural_video_ready":False}

if __name__=="__main__":
    import os
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working",type=Path,required=True)
    args=parser.parse_args()
    print("STUDIO_QUEUED_KERNEL_COMPATIBILITY_FIX",
          json.dumps(patch_queued_once(args.working),sort_keys=True))
