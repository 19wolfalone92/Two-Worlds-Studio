"""Kaggle private two-character canon proof; no video and no public source art."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path

from studio.kaggle_api import ROOT
from studio.kaggle_identity import HANDLE
from studio.release_gate import sha256

WORKER = "tools/kaggle_eb01_canon_entry.py"
MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
ARCHIVE = "EB01_CANON_SOURCE_PRIVATE.zip"
RECEIPT = "EB01_CANON_JOB.json"
CONTACT = "EB01_CANON_REVIEW_CONTACT.jpg"
SEEDS = {"KOSTYA": 920261112, "ELIZABETH": 920261113}

def prepare(username: str, nonce: str, destination: Path, root: Path = ROOT) -> dict:
    if not HANDLE.fullmatch(username):
        raise ValueError("Invalid private Kaggle username")
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,90}", nonce):
        raise ValueError("Nonce required for private audited image job")
    root = root.resolve()
    worker = (root / WORKER).resolve()
    if root not in worker.parents or not worker.is_file():
        raise ValueError("Canon worker source is missing")
    source = worker.read_text(encoding="utf-8")
    if source.count("__EB01_CANON_CONFIG_B64__") != 1:
        raise ValueError("Canon worker source has an unexpected template")
    revision = os.environ.get("GITHUB_SHA", "local-test-revision")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{8,80}", revision):
        raise ValueError("Unexpected Git source identifier")
    payload = {
        "schema":1,"nonce":nonce,"source_revision":revision,"model":MODEL,
        "seeds":SEEDS,"max_images":2,"allow_video":False,"private":True}
    slug = "eb01-v6-canon-" + hashlib.sha256(nonce.encode()).hexdigest()[:12]
    metadata = {
        "id":username+"/"+slug, "title":slug,
        "code_file":"kernel_main.py", "language":"python", "kernel_type":"script",
        "is_private":"true", "enable_gpu":"true", "enable_internet":"true",
        "machine_shape":"NvidiaTeslaT4","dataset_sources":[],
        "competition_sources":[],"kernel_sources":[],"model_sources":[]}
    dest=Path(destination)
    dest.mkdir(parents=True,exist_ok=False)
    rendered=source.replace("__EB01_CANON_CONFIG_B64__",
      base64.b64encode(json.dumps(payload,sort_keys=True).encode()).decode())
    (dest/"kernel_main.py").write_text(rendered,encoding="utf-8")
    (dest/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    return {
      "status":"PRIVATE_ONE_PASS_CANON_JOB_PREPARED","kernel":metadata["id"],
      "nonce":nonce,"model":MODEL,"max_images":2,"output_private":True,
      "gpu_executed":False,"publish":False,
      "worker_source_sha256":hashlib.sha256(source.encode()).hexdigest()
    }

def verify(root: Path, nonce: str) -> dict:
    root=Path(root)
    matches=list(root.rglob(RECEIPT))
    archives=list(root.rglob(ARCHIVE))
    previews=list(root.rglob(CONTACT))
    if len(matches)!=1 or len(archives)!=1 or len(previews)!=1:
        raise FileNotFoundError("Exactly one original archive+receipt+review contact required")
    receipt=json.loads(matches[0].read_text(encoding="utf-8"))
    if not (receipt.get("success") is True and receipt.get("nonce")==nonce
            and receipt.get("model")==MODEL and receipt.get("private") is True
            and receipt.get("gpu_inference_performed") is True
            and receipt.get("video_generated") is False
            and receipt.get("published") is False
            and receipt.get("image_identity_verified") is False
            and receipt.get("full_video_allowed") is False):
        raise ValueError("Not valid real two-character private output")
    if sha256(archives[0]) != receipt.get("archive_sha256"):
        raise ValueError("Original private PNG archive checksum mismatch")
    if sha256(previews[0]) != receipt.get("contact_sha256"):
        raise ValueError("Reviewer contact checksum mismatch")
    if not 18_000 < previews[0].stat().st_size < 2_000_000:
        raise ValueError("Invalid review-only contact image size")
    with zipfile.ZipFile(archives[0]) as z:
        if set(z.namelist())!={"canon/KOSTYA.png","canon/ELIZABETH.png"}:
            raise ValueError("Only two original canonical candidates are allowed")
        if set(receipt.get("generated_images",{}))!={"KOSTYA","ELIZABETH"}:
            raise ValueError("Two distinct character images are required")
        for key in ("KOSTYA","ELIZABETH"):
            original=z.read("canon/"+key+".png")
            row=receipt["generated_images"][key]
            if (not original.startswith(b"\x89PNG\r\n\x1a\n")
                    or len(original)!=row.get("size")
                    or hashlib.sha256(original).hexdigest()!=row.get("sha256")
                    or row.get("width")!=704 or row.get("height")!=1216
                    or row.get("art_reviewed") is not False):
                raise ValueError("Canonical image bytes are not source-locked")
    return {
       "status":"TWO_PRIVATE_REAL_CANDIDATES_REQUIRE_HUMAN_REVIEW",
       "nonce":nonce, "private_kernel":True,
       "images":{k:{"sha256":v["sha256"],"bytes":v["size"]}
                 for k,v in receipt["generated_images"].items()},
       "review_contact_sha256":receipt["contact_sha256"],
       "gpu_inference_performed":True, "video_generated":False,
       "visual_identity_approved":False,"published":False}

def main(argv=None):
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("prepare")
    a.add_argument("--username",required=True)
    a.add_argument("--nonce",required=True)
    a.add_argument("--destination",type=Path,required=True)
    b=sub.add_parser("verify")
    b.add_argument("--path",type=Path,required=True)
    b.add_argument("--nonce",required=True)
    x=p.parse_args(argv)
    r=prepare(x.username,x.nonce,x.destination) if x.action=="prepare" else verify(x.path,x.nonce)
    print(json.dumps(r,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
