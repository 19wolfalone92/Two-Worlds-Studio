"""Kaggle CPU-only probe for three PRIVATE completed/failed notebook outputs.

Verifies the original art and six WAV mounts before triggering any video GPU.
Never uploads originals or writes copies to public GitHub.
"""
from __future__ import annotations
import hashlib
import json
import zipfile
from pathlib import Path

OUTPUT=Path("/kaggle/working/EB01_V7_MOUNT_READONLY.json")
ROOT=Path("/kaggle/input")
WAV_SHA={
  "S01":"58763966f8084cafd7a82221e7846c0f57b1ffe5234c4905afc4ce1691c5bc26",
  "S02":"d4be318dbe660ade3a0a0289a482244f5b336a66001f0bdfc1c16b5051964b33",
  "S03":"21c5bffae7bccdd179365cd0d139222d2ed484b74b7e7e792937b31d1d895296",
  "S04":"6fd367739292a710ade59887d10b2d7ec19a67d563eec19cbd2c1aff355f1965",
  "S05":"a047e6ecedbeb33524f41e37ac61e2d7785b9051c669318460edbbc6a37cd5e8",
  "S06":"a9e29b7fa792e840d8c9fc02775a119c52ea1527bc9d96624506b7663c69ce21",
}
KOSTYA_SHA="447373a1ee3162dd81ab29e07f5cbff24dbe4bab44ddbbc7394213ecb6ca0afd"
ELIZABETH_SHA="7c691765a9f7cf3c190d110f114ce7d3b8a2f39eb681ecfe3056817c8e9a58a1"

def probe():
    result={"schema":1,"status":"NOT_READY","private_kernel_mounts":True,
            "gpu_executed":False,"video_generated":False,"published":False}
    roots=sorted(p.name for p in ROOT.iterdir()) if ROOT.exists() else []
    result["mounted_names"]=roots
    found={}
    for name in ("EB01_CANON_SOURCE_PRIVATE.zip","EB01_ELIZABETH_PRIVATE_ONLY.zip"):
        options=list(ROOT.rglob(name)) if ROOT.exists() else []
        if len(options)!=1:
            result["reason"]="Missing/ambiguous private mounted original: "+name
            return result
        found[name]=options[0]
    with zipfile.ZipFile(found["EB01_CANON_SOURCE_PRIVATE.zip"]) as z:
        old=z.read("canon/KOSTYA.png")
    with zipfile.ZipFile(found["EB01_ELIZABETH_PRIVATE_ONLY.zip"]) as z:
        photo=z.read("canon/ELIZABETH.png")
    if hashlib.sha256(old).hexdigest()!=KOSTYA_SHA or hashlib.sha256(photo).hexdigest()!=ELIZABETH_SHA:
        result["reason"]="Private source picture does not match approved SHA"
        return result
    result["character_sha256"]={"KOSTYA":KOSTYA_SHA,"ELIZABETH":ELIZABETH_SHA}
    wav={}
    for sid,sha in WAV_SHA.items():
        options=list(ROOT.rglob(sid+".wav")) if ROOT.exists() else []
        good=[p for p in options if hashlib.sha256(p.read_bytes()).hexdigest()==sha]
        if len(good)!=1:
            result["reason"]="Cannot mount source-pinned Russian WAV "+sid
            result["file_candidates"]=len(options)
            return result
        wav[sid]={"sha256":sha,"bytes":good[0].stat().st_size}
    result["voices"]=wav
    result["status"]="ALL_THREE_PRIVATE_OUTPUTS_MOUNTED_AND_SHA_VERIFIED"
    result["no_public_source_media"]=True
    return result

if __name__=="__main__":
    try:
        status=probe()
        OUTPUT.write_text(json.dumps(status,indent=2,ensure_ascii=False)+"\n")
        print("V7_READONLY_MOUNT_STATUS",status["status"],status.get("reason",""),flush=True)
        if status["status"]!="ALL_THREE_PRIVATE_OUTPUTS_MOUNTED_AND_SHA_VERIFIED":
            raise SystemExit(2)
    except BaseException as e:
        if not OUTPUT.exists():
            OUTPUT.write_text(json.dumps({"status":"MOUNT_PROBE_EXCEPTION",
              "exception":type(e).__name__,"gpu_executed":False,"published":False})+"\n")
        raise
