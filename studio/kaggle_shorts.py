"""Offline packaging and public-safe verification for a PRIVATE Kaggle T4x2 run.

No secrets, story text, assets or model weights enter the generated kernel code.
All private materials are mounted via authorized private Kaggle datasets.
The `prepare` command does not start a run or consume GPU quota.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

SOURCES = ("studio/__init__.py", "studio/offline_media.py",
           "studio/autonomous_factory.py", "studio/shorts_factory.py",
           "studio/kaggle_shorts_worker.py")
ALLOWED_PHASES = {"preflight", "image", "voice", "music", "render"}
DATASET = re.compile(r"^[A-Za-z0-9_-]{2,30}/[a-z0-9][a-z0-9-]{1,70}$")
USERNAME = re.compile(r"^[A-Za-z0-9_-]{2,30}$")
SLUG = re.compile(r"^[a-z][a-z0-9-]{2,60}$")
SPEAKERS = {"Serena", "Ryan", "Vivian", "Aiden", "Sohee", "Ono_Anna", "Dylan", "Eric", "Uncle_Fu"}


def build_source(root: Path) -> bytes:
    root = Path(root).resolve()
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in SOURCES:
            src = (root / rel).resolve()
            if root not in src.parents or not src.is_file() or src.stat().st_size > 220_000:
                raise FileNotFoundError(f"Missing/oversized audited source: {rel}")
            z.write(src, rel)
    if data.tell() > 1_000_000:
        raise ValueError("Source archive unusually large")
    return data.getvalue()


def build_kernel(root: Path, destination: Path, *, username: str, dataset: str,
                 phase: str, story: str, first_scene: int = 1, max_scenes: int = 2,
                 gpu_id: int = 0, budget_s: int = 2400, speaker: str = "Serena",
                 slug: str = "two-worlds-russian-short-gpu",
                 source_path: str = "/kaggle/input/two-worlds-private-queue/Two_Worlds_Studio_RU_Batch03_Ready_For_Production.zip",
                 model_path: str = "/kaggle/input/two-worlds-models",
                 recovery_dataset: str | None = None, recovery_path: str | None = None,
                 model_dataset: str | None = None,
                 install_deps: bool = False, bootstrap: bool = False) -> dict:
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    if not USERNAME.fullmatch(username) or not SLUG.fullmatch(slug):
        raise ValueError("Invalid Kaggle owner/slug")
    if (not DATASET.fullmatch(dataset) or (recovery_dataset and not DATASET.fullmatch(recovery_dataset))
            or (model_dataset and not DATASET.fullmatch(model_dataset))):
        raise ValueError("Invalid Kaggle dataset handle")
    if phase not in ALLOWED_PHASES or story not in {str(i) for i in range(13, 19)}:
        raise ValueError("Unknown project phase or story")
    if not (1 <= first_scene <= 8 and 1 <= max_scenes <= 8):
        raise ValueError("Choose 1..8 scenes per run")
    if gpu_id not in (0, 1) or not 60 <= budget_s <= 18000:
        raise ValueError("Invalid Kaggle GPU index/time budget")
    if speaker not in SPEAKERS:
        raise ValueError("Unsupported Qwen speaker")
    # Only read local source/attached Kaggle path, never accept arbitrary code URLs.
    if not source_path.startswith("/kaggle/input/") or ".." in Path(source_path).parts:
        raise ValueError("Private story source must be under /kaggle/input")
    if not (model_path.startswith("/kaggle/working/") if bootstrap else
            model_path.startswith("/kaggle/input/")) or ".." in Path(model_path).parts:
        raise ValueError("Model path must point to an attached dataset, or writable work for bootstrap")
    if recovery_path and (not recovery_dataset or not recovery_path.startswith("/kaggle/input/")
                          or ".." in Path(recovery_path).parts):
        raise ValueError("Recovery ZIP requires an attached private recovery dataset")
    if bootstrap and phase not in {"image", "voice"}:
        raise ValueError("Model bootstrap is only for generation phases")
    if bootstrap and not install_deps:
        raise ValueError("Bootstrap needs explicitly enabled dependency installation")
    if phase in ("image", "voice") and not bootstrap and model_dataset is None:
        raise ValueError("Generation requires a private mounted model dataset or explicit bootstrap")
    template = (root / "tools/kaggle_shorts_kernel_entry.py").read_text(encoding="utf-8")
    payload = {
        "story": story, "phase": phase, "first_scene": first_scene,
        "max_scenes": max_scenes, "gpu_id": gpu_id, "budget_s": budget_s,
        "speaker": speaker, "source_path": source_path, "model_path": model_path,
        "recovery_path": recovery_path, "install_deps": install_deps,
        "bootstrap": bootstrap, "publication_approved": False,
    }
    bundle = build_source(root)
    if template.count("__TW_SHORTS_JOB_B64__") != 1 or template.count("__TW_SHORTS_SOURCE_B64__") != 1:
        raise ValueError("Invalid kernel template placeholders")
    entry = template.replace("__TW_SHORTS_JOB_B64__",
                             base64.b64encode(json.dumps(payload).encode()).decode())
    entry = entry.replace("__TW_SHORTS_SOURCE_B64__", base64.b64encode(bundle).decode())
    meta = {"id": f"{username}/{slug}", "title": "Two Worlds Russian Story GPU",
            "code_file": "kernel_main.py", "language": "python",
            "kernel_type": "script", "is_private": "true",
            "enable_gpu": "true" if phase in {"image", "voice"} else "false",
            "enable_internet": "true" if (install_deps or bootstrap) else "false",
            "machine_shape": "NvidiaTeslaT4" if phase in {"image", "voice"} else "",
            "dataset_sources": [dataset] + ([model_dataset] if model_dataset else []) +
                               ([recovery_dataset] if recovery_dataset else []),
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    destination.mkdir(parents=True,exist_ok=True)
    for filename in ("kernel_main.py","kernel-metadata.json"):
        if (destination/filename).exists():
            raise FileExistsError("Refusing to overwrite an existing Kaggle kernel package")
    (destination / "kernel_main.py").write_text(entry,encoding="utf-8")
    (destination / "kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    return {"kernel":meta["id"],"private":True,"phase":phase,
            "story":story,"gpu_requested":meta["enable_gpu"]=="true",
            "gpu_memory_pooled":False,"source_sha256":hashlib.sha256(bundle).hexdigest(),
            "source_bytes":len(bundle),"story_script_embedded":False,
            "credentials_embedded":False,"output":str(destination),
            "generation_started":False,"dataset_sources":meta["dataset_sources"]}


def verify_result(output: Path, story: str) -> dict:
    output = Path(output).resolve()
    receipt_file = output / "two_worlds_kaggle_receipt.json"
    if not receipt_file.is_file():
        raise FileNotFoundError("Actual Kaggle receipt not found")
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if receipt.get("reviewed") is not False or receipt.get("published") is not False:
        raise ValueError("Unexpected fake review/publication flag")
    if "asset_bundle" not in receipt:
        raise ValueError("No real private output bundle returned")
    bundle = receipt["asset_bundle"]
    zipfile_path = output / Path(bundle["zip"]).name
    if not zipfile_path.is_file():
        raise FileNotFoundError("Private output ZIP missing")
    if hashlib.sha256(zipfile_path.read_bytes()).hexdigest()!=bundle.get("sha256"):
        raise ValueError("Kaggle output ZIP SHA-256 mismatch")
    with zipfile.ZipFile(zipfile_path) as z:
        names=z.namelist()
        if not any(n==f"production_queue/{story}/project.json" for n in names):
            raise ValueError("Output is not for requested story")
        for name in names:
            pure=Path(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError("Unsafe Kaggle output ZIP")
    return {"kaggle_success":receipt.get("success") is True,
            "output_sha256":bundle["sha256"],"files":len(names),
            "ready_to_publish":False,"story":story}


def main(argv=None):
    p=argparse.ArgumentParser(description="Build offline Kaggle T4x2 private kernel safely")
    cmd=p.add_subparsers(dest="action",required=True)
    b=cmd.add_parser("prepare")
    b.add_argument("--username",required=True)
    b.add_argument("--private-dataset",required=True)
    b.add_argument("--source-path",default="/kaggle/input/two-worlds-private-queue/Two_Worlds_Studio_RU_Batch03_Ready_For_Production.zip")
    b.add_argument("--model-path",default="/kaggle/input/two-worlds-models")
    b.add_argument("--story",choices=[str(i) for i in range(13,19)],required=True)
    b.add_argument("--phase",choices=sorted(ALLOWED_PHASES),default="preflight")
    b.add_argument("--start",type=int,default=1)
    b.add_argument("--count",type=int,default=2)
    b.add_argument("--gpu-id",type=int,choices=(0,1),default=0)
    b.add_argument("--budget",type=int,default=2400)
    b.add_argument("--speaker",default="Serena")
    b.add_argument("--model-dataset")
    b.add_argument("--recovery-dataset")
    b.add_argument("--recovery-path")
    b.add_argument("--install-deps",action="store_true")
    b.add_argument("--bootstrap",action="store_true")
    b.add_argument("--out",type=Path,required=True)
    b.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parent.parent)
    v=cmd.add_parser("verify")
    v.add_argument("--out",type=Path,required=True)
    v.add_argument("--story",required=True)
    args=p.parse_args(argv)
    if args.action=="prepare":
        result=build_kernel(args.source_root,args.out,username=args.username,dataset=args.private_dataset,
                            phase=args.phase,story=args.story,first_scene=args.start,max_scenes=args.count,
                            gpu_id=args.gpu_id,budget_s=args.budget,speaker=args.speaker,
                            source_path=args.source_path,model_path=args.model_path,
                            recovery_dataset=args.recovery_dataset,recovery_path=args.recovery_path,
                            model_dataset=args.model_dataset,
                            install_deps=args.install_deps,bootstrap=args.bootstrap)
    else:
        result=verify_result(args.out,args.story)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()