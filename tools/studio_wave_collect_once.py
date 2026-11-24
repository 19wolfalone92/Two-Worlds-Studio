"""Idempotent readback of the SAME private Kaggle Studio native video job.

No GPU requests or kernel pushes. Queue/RUNNING is a pending state,
not a fabricated model error. Only real Kaggle COMPLETE plus verified
model-native MP4 is eligible for encrypted transport to the owner.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from studio.sealed_artifacts import seal
from tools.studio_wave_status_gate import (
    JobStateError,KERNEL,SOURCE_SHA256,PENDING,FAILURE,
    parse_state,classify,verify_actual_native_mp4,VIDEO_NAME,
)

def _cli(args:list[str],timeout:int)->str:
    try:
        result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    except (OSError,subprocess.TimeoutExpired) as exc:
        raise JobStateError("KAGGLE_API_NOT_RESPONDING_"+type(exc).__name__) from exc
    if result.returncode!=0:
        # Never print private stderr (can contain prompts, local paths or tokens).
        raise JobStateError("KAGGLE_CLI_RETURNED_NONZERO")
    return result.stdout

def collect_once(root:Path,*,status_fn=None,output_fn=None,
                 verify_fn=verify_actual_native_mp4,seal_fn=seal)->dict:
    """Unit-testable pure state transitions with real protected output path."""
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    state=parse_state(
        status_fn() if status_fn is not None else
        _cli(["kaggle","kernels","status",KERNEL],65))
    snapshot=classify(state)
    result={"schema":1,"kernel":KERNEL,"kernel_state":state,**snapshot,
            "new_gpu_jobs_submitted":0,"private_source_published":False,
            "human_artistic_approved":False,"published":False}
    if state in PENDING:
        # Not even one Kaggle output request or GPU submit.
        return result
    if state in FAILURE:
        result["requires_private_failure_forensic"]=True
        return result
    media=root/"original_private_output"
    media.mkdir(mode=0o700,exist_ok=True)
    if output_fn is None:
        _cli(["kaggle","kernels","output",KERNEL,
              "-p",str(media),"--force"],240)
    else:
        output_fn(media)
    proof=verify_fn(media)
    if (proof.get("status")!="TRUE_NEURAL_MP4_TECHNICALLY_VERIFIED_OWNER_REVIEW_REQUIRED"
            or proof.get("human_artistic_approved") is not False):
        raise JobStateError("Final native video proof was not approved by verification gate")
    recipient=Path("ops/studio_wave_native_recipient_public.pem")
    if not recipient.is_file() and seal_fn is seal:
        raise JobStateError("Missing recipient public key; private movie cannot leave runner")
    private_ciphertext=root/"STUDIO_WAVE_NATIVE_USER_ENCRYPTED.twenc"
    report=seal_fn(media/VIDEO_NAME,recipient,private_ciphertext,
                   {"purpose":"studio-native-wave-output","schema":1,
                    "source_sha256":SOURCE_SHA256})
    result.update(status="TRUE_GPU_MP4_VERIFIED_ENCRYPTED_OWNER_REVIEW",
                  real_mp4_accepted=True,
                  verified_original_frames=proof["frames"],
                  raw_mp4_sha256=proof["sha256"],
                  ciphertext_sha256=report["sha256"],
                  private_result_encrypted=True,
                  needs_owner_artistic_review=True)
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working",type=Path,required=True)
    args=parser.parse_args(argv)
    work=args.working
    work.mkdir(parents=True,exist_ok=True)
    result=collect_once(work)
    # Safe numeric summary only; raw source, video and Kaggle logs are private.
    (work/"STUDIO_WAVE_SAFE_STATE.json").write_text(json.dumps(result,indent=2)+"\n")
    print("STUDIO_NATIVE_WAVE_STATE",json.dumps(result,sort_keys=True))
    if result["status"]=="KAGGLE_TERMINAL_FAILURE":
        raise SystemExit(2)

if __name__=="__main__":
    main()
