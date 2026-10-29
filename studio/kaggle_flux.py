"""Optional Kaggle FLUX.2 klein 4B image generation — explicit GPU execution only.

No image generation happens in GitHub CI. Validates planned jobs in --dry-run,
supports ORIGINAL character references or scene prompts, and writes only
unreviewed PNG assets with traceable metadata.

Model: black-forest-labs/FLUX.2-klein-4B (Apache-2.0 per official model card).
T4/P100 have NO bf16, so use float16 + CPU offloading. Running this on Kaggle is
experimental: memory quota, downloads and unsupported kernels can still fail.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from studio.art_director import ROOT,load_style

DEFAULT_MODEL="black-forest-labs/FLUX.2-klein-4B"
VIEWS={"front","three_quarter","profile","full_body","expression","back","rear_three_quarter"}
# Keep the original five view seeds stable when new angles are introduced.
VIEW_SEED_OFFSET={"expression":0,"front":1,"full_body":2,"profile":3,"three_quarter":4,
                  "back":5,"rear_three_quarter":6}
SAFE_ID=re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,32}$")
VIEWHINT={
    "front":"neutral direct FRONT VIEW, level camera, shoulders squared, face fully visible",
    "three_quarter":"three-quarter 45 degree portrait, distinct facial anatomy unchanged",
    "profile":"strict SIDE PROFILE, same person, consistent facial bone structure",
    "full_body":"full-body standing turnaround, correct hands and proportion",
    "expression":"cinematic expressive portrait, emotionally restrained, accurate face",
    "back":"strict BACK view, back of head and clothes, no visible face, correct body proportions",
    "rear_three_quarter":"REAR THREE-QUARTER view, same body, facing away from camera, never horizontally flip"
}

def sha256(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_reference(path:Path|None)->dict|None:
    if path is None:
        return None
    if not path.is_file() or path.suffix.lower() not in (".png",".jpg",".jpeg",".webp"):
        raise ValueError("Reference must be a real PNG/JPEG/WebP file")
    if path.stat().st_size>25*1024*1024:
        raise ValueError("Reference image exceeds 25 MiB")
    from PIL import Image
    with Image.open(path) as im:
        im.verify()
    return {"path":str(path),"sha256":sha256(path)}

def character_job(style:dict,body:str,view:str,reference:Path|None=None)->dict:
    if body not in style["characters"]:
        raise ValueError(f"Unknown physical character body {body}")
    if view not in VIEWS:
        raise ValueError(f"Unknown turnaround view {view}")
    ch=style["characters"][body]
    location="Earth" if body.startswith("earth_") else "Noctaris"
    prompt=(
        f"{style['visual_style']} "
        f"Premium original graphic novel character model sheet, visually coherent, "
        f"vertical 9:16, professional dramatic portrait, no printed text, no logos. "
        f"Physical identity: {ch['body_name']}, age {ch['age']}, {ch['body']}. "
        f"Fixed distinctive details: {ch['distinctive']}. "
        f"Clothing: {ch['clothing']}. Home reality: {location}. "
        f"Composition: {VIEWHINT[view]}. "
        "Character must look unmistakably identical to the supplied reference when present. "
        "Clean anatomy, real clothing materials, fine textures, organic shadows."
    )
    return {
        "job_kind":"character","body":body,"view":view,
        "prompt":prompt,"negative":style["negative"],
        # Same identity seed across all views is useful but NOT proof of identity.
        "seed":int(ch["seed_anchor"])+VIEW_SEED_OFFSET[view],
        "reference":validate_reference(reference),
        "review_status":"unreviewed","visual_identity_verified":False
    }

def scene_job(pack:dict,scene_id:str,reference:Path|None=None)->dict:
    if not isinstance(pack.get("scenes"),list) or not SAFE_ID.fullmatch(pack.get("episode","")):
        raise ValueError("Invalid art prompt-pack JSON")
    match=[s for s in pack["scenes"] if s.get("scene")==scene_id]
    if len(match)!=1:
        raise ValueError("Scene not found exactly once in Work/Studio prompt pack")
    item=match[0]
    if not isinstance(item.get("prompt"),str) or len(item["prompt"])<40:
        raise ValueError("Missing detailed scene art prompt")
    if not isinstance(item.get("seed"),int) or not 0<=item["seed"]<2**32:
        raise ValueError("Invalid saved art seed")
    return {
        "job_kind":"scene","episode":pack["episode"],"scene":scene_id,
        "prompt":item["prompt"],"negative":item.get("negative",""),
        "seed":item["seed"],"bodies":item.get("bodies",[]),
        "reference":validate_reference(reference),
        "review_status":"unreviewed","visual_identity_verified":False
    }

def plan_job(args)->dict:
    if getattr(args,"character_project",None):
        from studio.character_art import get_job,resolve_asset
        if not getattr(args,"job",None):
            raise ValueError("--character-project requires --job from that project's jobs.json")
        root=Path(args.character_project)
        source=get_job(root,args.job)
        if len(source["body_ids"])!=1 or source.get("edit_target"):
            raise ValueError("FLUX worker currently supports one physical identity and no separate edit target")
        ref=source["references"][0]
        return {"job_kind":"character","body":source["body_ids"][0],"view":source["view"],
                "prompt":source["prompt"],"negative":"duplicate anatomy, watermark, malformed eyes",
                "seed":int(source["job_fingerprint"][:8],16),
                "reference":validate_reference(resolve_asset(root,ref["file"])),
                "source_job_fingerprint":source["job_fingerprint"],
                "reference_mode":"single_primary_identity_reference",
                "review_status":"unreviewed","visual_identity_verified":False}
    style=load_style()
    reference=Path(args.reference) if args.reference else None
    if args.mode=="character":
        if not args.body:
            raise ValueError("--mode character requires --body")
        return character_job(style,args.body,args.view,reference)
    if not args.prompt_pack or not args.scene:
        raise ValueError("--mode scene requires --prompt-pack and --scene")
    data=json.loads(Path(args.prompt_pack).read_text(encoding="utf-8"))
    return scene_job(data,args.scene,reference)

def run_gpu(job:dict,output:Path,*,model_id:str=DEFAULT_MODEL,
            width:int=576,height:int=1024,steps:int=4)->dict:
    if model_id!=DEFAULT_MODEL:
        raise ValueError("Only the reviewed Apache-2.0 FLUX.2 klein 4B checkpoint is allowed")
    if (width,height)!=(576,1024) and (width%64 or height%64 or
                                 min(width,height)<448 or max(width,height)>1344):
        raise ValueError("Use 576x1024 or safe multiples of 64 between 448 and 1344")
    if steps<1 or steps>8:
        raise ValueError("Klein distilled sampling uses 1..8 steps")
    if job.get("reference"):
        current=validate_reference(Path(job["reference"]["path"]))
        if current["sha256"]!=job["reference"].get("sha256"):
            raise ValueError("Identity reference changed after drawing job compilation")
    try:
        import torch
        from diffusers import Flux2KleinPipeline
        from diffusers.utils import load_image
    except ImportError as exc:
        raise RuntimeError("Install Kaggle FLUX dependencies from the Notebook first") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU. Enable Kaggle Notebook Settings > Accelerator > GPU.")
    device_name=torch.cuda.get_device_name(0)
    # NVIDIA T4 and P100 do not support BF16; dtype based on actual CUDA support.
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    gpu_gb=torch.cuda.get_device_properties(0).total_memory/(1024**3)
    print(f"Selected {device_name} ({gpu_gb:.1f} GiB), precision={dtype}; model loading may be slow.")
    try:
        pipe=Flux2KleinPipeline.from_pretrained(model_id,torch_dtype=dtype)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=False)
        kwargs={
            "prompt":job["prompt"],"width":width,"height":height,
            "num_inference_steps":steps,"guidance_scale":1.0,
            "generator":torch.Generator(device="cpu").manual_seed(job["seed"])
        }
        if job["reference"]:
            if sha256(Path(job["reference"]["path"]))!=job["reference"]["sha256"]:
                raise ValueError("Identity reference changed while loading the model")
            kwargs["image"]=load_image(job["reference"]["path"])
        generated=pipe(**kwargs).images[0]
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("GPU memory insufficient: restart Kaggle, use 448x768 or 512x768 "
                           "and one image at a time. Loading on Kaggle T4 is not guaranteed.") from exc
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Image exists — refusing overwrite: {output}")
    generated.save(output,"PNG")
    if output.stat().st_size<10000:
        output.unlink(missing_ok=True)
        raise RuntimeError("Generated PNG unexpectedly small")
    return {"image":str(output),"sha256":sha256(output),"model":model_id,
            "gpu_name":device_name,"precision":str(dtype),
            "width":generated.width,"height":generated.height,
            "reference":job["reference"],
            "review_status":"unreviewed","identity_verified":False}

def execute(args)->dict:
    job=plan_job(args)
    outdir=Path(args.output_dir)
    output=outdir/((f'{job["body"]}_{job["view"]}' if job["job_kind"]=="character"
                     else f'{job["episode"]}_{job["scene"]}')+".png")
    if args.dry_run:
        return {"dry_run":True,"would_write":str(output),
                "model":DEFAULT_MODEL,"job":job,"real_images_generated":0}
    info=run_gpu(job,output,model_id=DEFAULT_MODEL,width=args.width,
                 height=args.height,steps=args.steps)
    metadata={"schema":1,"kind":job["job_kind"],"job":job, "result":info,
              "image_is_human_approved":False,
              "creator_claim":"AI generated with optional source reference; requires human QA",
              "no_rights_claim":"Model Apache-2.0 does not certify rights to any reference image."}
    meta_path=output.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"image":str(output),"manifest":str(meta_path),
            "status":"generated_unreviewed"}

def main():
    p=argparse.ArgumentParser(description="FLUX.2 klein 4B: only run with explicitly enabled Kaggle GPU")
    p.add_argument("--mode",choices=("character","scene"),required=True)
    p.add_argument("--body",help="Physical body, e.g. earth_mira")
    p.add_argument("--view",choices=sorted(VIEWS),default="front")
    p.add_argument("--prompt-pack",type=Path)
    p.add_argument("--scene",help="Scene ID from a Work-approved art prompt pack")
    p.add_argument("--reference",help="Explicit image path to edit/reuse identity; keep rights")
    p.add_argument("--character-project",type=Path,help="Current private reference project instead of legacy fixture prose")
    p.add_argument("--job",help="Exact drawing job ID from that project")
    p.add_argument("--output-dir",default="/kaggle/working/two_worlds_art")
    p.add_argument("--width",type=int,default=576)
    p.add_argument("--height",type=int,default=1024)
    p.add_argument("--steps",type=int,default=4)
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    print(json.dumps(execute(args),indent=2,ensure_ascii=False))
if __name__=="__main__":
    main()
