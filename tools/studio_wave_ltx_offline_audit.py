"""CPU-only verification of the EXACT Studio LTX 2B source patch against pinned upstream.

Extracts the GPU worker's actual source-edit statement block via Python AST
and applies it to a throwaway checkout of real upstream code, not a fake
placeholder. The pip/weight-loading command is explicitly disabled.
No CUDA, model download, Kaggle quota, user image, or inference takes place.
"""
from __future__ import annotations

import ast
import hashlib
import os
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
WORKER=REPO/"tools/kaggle_studio_man_wave_native.py"
UPSTREAM_COMMIT="50f0aef1297574a4b29acbdb874e3a363a6d5f2e"

def sha(p:Path)->str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def apply_worker_patch_offline(upstream:Path)->dict:
    upstream=Path(upstream).resolve()
    if not upstream.is_dir():
        raise ValueError("Pinned real LTX upstream clone is missing")
    native=ast.parse(WORKER.read_text(encoding="utf-8"),filename=str(WORKER))
    funcs=[x for x in native.body if isinstance(x,ast.FunctionDef) and x.name=="main"]
    if len(funcs)!=1:
        raise AssertionError("Native GPU worker main changed")
    body=funcs[0].body
    def find_assign(name):
        return [i for i,n in enumerate(body) if isinstance(n,ast.Assign) and
                any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    start=find_assign("source");end=find_assign("generated")
    if len(start)!=1 or len(end)!=1 or start[0]>=end[0]:
        raise AssertionError("Cannot locate exact production source patch in native Studio worker")
    patch_statements=body[start[0]:end[0]]
    text=WORKER.read_text()
    if text.count("from studio.eb01_native_t4_profile import patch_native_sources")<1:
        raise AssertionError("Native T4 reference VAE patch dependency lost")
    def no_install(args,timeout=0,cwd=None):
        if (len(args)>=3 and list(map(str,args[:3]))==
                [sys.executable,"-m","pip"]):
            return "OFFLINE_INSTALL_BLOCKED_BY_DESIGN"
        raise AssertionError("Unexpected external process during static Studio patch test")
    locals_for_patch={"LTX":upstream,"native_profile":True,
        "sha":sha,"run":no_install,"sys":sys,
        "hashlib":hashlib,"ROOT":REPO}
    executable=compile(ast.Module(body=patch_statements,type_ignores=[]),
                       filename=str(WORKER),mode="exec")
    exec(executable,locals_for_patch,locals_for_patch)
    paths=[
        upstream/"ltx_video"/"inference.py",
        upstream/"ltx_video"/"pipelines"/"pipeline_ltx_video.py",
        upstream/"ltx_video"/"models"/"autoencoders"/"vae.py",
    ]
    for p in paths:
        ast.parse(p.read_text(),filename=str(p))
    cfg=upstream/"configs"/"ltxv-2b-0.9.6-distilled.yaml"
    if "prompt_enhancement_words_threshold: 0" not in cfg.read_text():
        raise AssertionError("Prompt enhancer 3B was not disabled in offline patch")
    return {"schema":1,"status":"PINNED_OFFICIAL_LTX_2B_PATCH_AST_PASS",
            "gpu_executed":False,"model_downloaded":False,
            "user_image_processed":False,
            "official_upstream_git_sha":UPSTREAM_COMMIT,
            "patched_python_modules":len(paths),
            "source_sha256":{p.relative_to(upstream).as_posix():sha(p) for p in paths}}

if __name__=="__main__":
    import json
    result=apply_worker_patch_offline(Path(os.environ["STUDIO_LTX_UPSTREAM_DIR"]))
    print("STUDIO_OFFICIAL_LTX_NO_GPU_SOURCE_CONTRACT",json.dumps(result,sort_keys=True))
