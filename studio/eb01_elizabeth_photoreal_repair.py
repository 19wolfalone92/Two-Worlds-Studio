"""Private single photoreal Elizabeth repair, preserving existing KOSTYA PNG hash.

Kostya original is read from an EXISTING completed PRIVATE Kaggle output on an
ephemeral GitHub runner and embedded ONLY in a newly submitted PRIVATE worker,
never committed to public GitHub or exposed in Actions logs.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path

from studio.eb01_canon_kaggle import (ARCHIVE, RECEIPT, MIXED_MODEL,
                                    MODEL as ORIGINAL_MODEL, verify as verify_canon)
from studio.kaggle_api import ROOT
from studio.kaggle_identity import HANDLE

WORKER="tools/kaggle_eb01_elizabeth_repair_entry.py"
MODEL="RunDiffusion/Juggernaut-XL-v9"

def original_kostya(root: Path, nonce: str) -> tuple[bytes,str]:
    source=verify_canon(root,nonce)
    if source.get("status")!="TWO_PRIVATE_REAL_CANDIDATES_REQUIRE_HUMAN_REVIEW":
        raise ValueError("Existing canonical images were not genuinely produced")
    with zipfile.ZipFile(next(Path(root).rglob(ARCHIVE))) as z:
        bytes_=z.read("canon/KOSTYA.png")
    digest=hashlib.sha256(bytes_).hexdigest()
    if digest!=source["images"]["KOSTYA"]["sha256"] or len(bytes_)<100000:
        raise ValueError("Private skeleton original changed")
    return bytes_,digest

def prepare(username: str, nonce: str, previous_folder: Path,
            previous_nonce: str, destination: Path, root: Path = ROOT) -> dict:
    if not HANDLE.fullmatch(username) or not re.fullmatch("[A-Za-z0-9_-]{6,90}",nonce):
        raise ValueError("Invalid private Kaggle account/nonce")
    raw,original_sha=original_kostya(previous_folder,previous_nonce)
    source=(Path(root)/WORKER).read_text(encoding="utf-8")
    if source.count("__EB01_ELIZABETH_REPAIR_CONFIG_B64__")!=1:
        raise ValueError("Unreviewed one-image repair worker source template")
    payload={
        "schema":1,"nonce":nonce,"model":MODEL,
        "source_revision":os.environ.get("GITHUB_SHA","local-test-revision"),
        "elizabeth_seed":820261112,"reused_kostya":True,
        "kostya_sha256":original_sha,
        "kostya_png_b64":base64.b64encode(raw).decode("ascii"),
        "max_new_images":1,"allow_video":False,"private":True}
    rendered=source.replace("__EB01_ELIZABETH_REPAIR_CONFIG_B64__",
        base64.b64encode(json.dumps(payload,sort_keys=True).encode()).decode("ascii"))
    slug="eb01-v6-photoreal-"+hashlib.sha256(nonce.encode()).hexdigest()[:12]
    metadata={
        "id":username+"/"+slug, "title":slug,
        "code_file":"kernel_main.py","language":"python","kernel_type":"script",
        "is_private":"true","enable_gpu":"true","enable_internet":"true",
        "machine_shape":"NvidiaTeslaT4","dataset_sources":[],
        "competition_sources":[],"kernel_sources":[],"model_sources":[]}
    dest=Path(destination)
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"kernel_main.py").write_text(rendered,encoding="utf-8")
    (dest/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return {
        "status":"ONE_NEW_PHOTOREAL_ELIZABETH_PRIVATE_JOB_PREPARED",
        "kernel":metadata["id"],"source_model":MODEL,
        "original_KOSTYA_sha256":original_sha,
        "new_images":1,"reused_good_existing_images":1,
        "video_generated":False,"gpu_executed":False,"published":False}

def verify(path: Path, nonce: str, original_KOSTYA_sha256: str) -> dict:
    result=verify_canon(path,nonce)
    receipts=list(Path(path).rglob(RECEIPT))
    if len(receipts)!=1:
        raise FileNotFoundError("Missing repaired source receipt")
    data=json.loads(receipts[0].read_text(encoding="utf-8"))
    if (data.get("model")!=MIXED_MODEL
            or data.get("parent_KOSTYA_sha256")!=original_KOSTYA_sha256
            or data["generated_images"]["ELIZABETH"].get("model")!=MODEL):
        raise ValueError("Targeted photoreal repair provenance is incorrect")
    return {
        **result,"status":"PHOTOREAL_ELIZABETH_REPAIRED_KOSTYA_PRESERVED",
        "source_model_KOSTYA":ORIGINAL_MODEL,
        "source_model_ELIZABETH":MODEL,
        "unchanged_KOSTYA_SHA256":original_KOSTYA_sha256,
        "human_review_required":True,"published":False}

def main(argv=None):
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("prepare")
    a.add_argument("--username",required=True)
    a.add_argument("--nonce",required=True)
    a.add_argument("--source-output",type=Path,required=True)
    a.add_argument("--source-nonce",required=True)
    a.add_argument("--destination",type=Path,required=True)
    b=sub.add_parser("verify")
    b.add_argument("--path",type=Path,required=True)
    b.add_argument("--nonce",required=True)
    b.add_argument("--parent-kostya-sha256",required=True)
    x=p.parse_args(argv)
    result=(prepare(x.username,x.nonce,x.source_output,x.source_nonce,x.destination)
            if x.action=="prepare" else verify(x.path,x.nonce,x.parent_kostya_sha256))
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
