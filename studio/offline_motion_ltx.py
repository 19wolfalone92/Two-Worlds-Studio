"""Opt-in, no-network LTX-Video 2B image-to-video launcher.

Runs ONLY with model weights and official LTX source already staged locally.
Does not download, publish, merge, charge API credits or fake video output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL="ltxv-2b-0.9.8-distilled"
CONFIG=MODEL+".yaml"
CKPT=MODEL+".safetensors"
UPSCALER="ltxv-spatial-upscaler-0.9.8.safetensors"


class MotionBlocked(RuntimeError):
    pass


def sha(path:Path)->str:
    out=hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda:file.read(1<<20),b""):
            out.update(block)
    return out.hexdigest()


def yaml_override(text:str,key:str,value:Path)->str:
    expr=r"(?m)^"+re.escape(key)+r":\s*[^\r\n]+$"
    if len(re.findall(expr,text))!=1:
        raise MotionBlocked(f"Unexpected reviewed model YAML key: {key}")
    return re.sub(expr,lambda _:f"{key}: {json.dumps(str(value),ensure_ascii=False)}",text)


def prepare(repo:Path,weights:Path,image:Path,output:Path,prompt:str,
            width:int=512,height:int=896,frames:int=81,seed:int=1729)->dict:
    repo=Path(repo).resolve(); weights=Path(weights).resolve()
    image=Path(image).resolve(); output=Path(output).resolve()
    if not isinstance(prompt,str) or not 50<=len(prompt)<=1500:
        raise MotionBlocked("Specify 50-1500 characters of actual actions/camera motion")
    if width%32 or height%32 or min(width,height)<320 or max(width,height)>1280:
        raise MotionBlocked("Dimensions must be multiples of 32, within 320-1280")
    if frames%8!=1 or not 17<=frames<=129:
        raise MotionBlocked("Frame count must be 8×N+1")
    if type(seed)!=int or not 0<=seed<2**32:
        raise MotionBlocked("Invalid seed")
    if image.suffix.lower() not in {".jpg",".jpeg",".png",".webp"}:
        raise MotionBlocked("Use a real licensed private reference image")
    if output==repo or repo in output.parents:
        raise MotionBlocked("Do not export personal reference media to public source repo")
    if output.exists() and any(output.iterdir()):
        raise MotionBlocked("Output directory must be empty (protect past results)")
    locations={"runner":repo/"inference.py","config":repo/"configs"/CONFIG,
               "checkpoint":weights/CKPT,"upscaler":weights/UPSCALER,
               "encoder":weights/"pixart-text-encoder"/"text_encoder"/"config.json",
               "tokenizer":weights/"pixart-text-encoder"/"tokenizer"/"tokenizer_config.json",
               "image":image}
    missing=[key for key,path in locations.items() if not path.is_file()]
    return {"model":MODEL,"state":"BLOCKED" if missing else "READY_FOR_GPU_SMOKE_TEST",
            "missing":missing,"paths":{key:str(p) for key,p in locations.items()},
            "source_sha256":sha(image) if image.is_file() else None,
            "dimensions":[width,height,frames],"seed":seed,"prompt":prompt,
            "output":str(output),"published":False,"real_gpu_test":False}


def execute(plan:dict,run_gpu:bool=False)->dict:
    if plan["missing"]:
        raise MotionBlocked(f"Missing locally staged files: {plan['missing']}")
    if not run_gpu:
        return {"state":"READY_FOR_GPU_SMOKE_TEST","generated":False,
                "downloaded":False,"credits_spent":0}
    try:
        import torch
    except ImportError as exc:
        raise MotionBlocked("PyTorch missing in user-controlled GPU runtime") from exc
    if not torch.cuda.is_available():
        raise MotionBlocked("No CUDA GPU; real video not generated")
    if torch.cuda.get_device_properties(0).total_memory<12*(1024**3):
        raise MotionBlocked("Conservative VRAM floor 12 GiB; model not proven on this GPU")
    files={key:Path(p) for key,p in plan["paths"].items()}
    output=Path(plan["output"]); output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="two-worlds-ltxv-") as temp:
        cfg=Path(temp)/"offline-config.yaml"
        value=files["config"].read_text(encoding="utf8")
        for key,path in (("checkpoint_path",files["checkpoint"]),
                         ("spatial_upscaler_model_path",files["upscaler"]),
                         ("text_encoder_model_name_or_path",files["encoder"].parent.parent)):
            value=yaml_override(value,key,path)
        cfg.write_text(value,encoding="utf8")
        w,h,frames=plan["dimensions"]
        args=[sys.executable,str(files["runner"]),"--prompt",plan["prompt"],
              "--conditioning_media_paths",str(files["image"]),
              "--conditioning_start_frames","0","--height",str(h),
              "--width",str(w),"--num_frames",str(frames),
              "--frame_rate","24","--seed",str(plan["seed"]),
              "--offload_to_cpu","true","--pipeline_config",str(cfg),
              "--output_path",str(output)]
        env=os.environ.copy()
        env.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
                    "HF_HUB_DISABLE_TELEMETRY":"1","TOKENIZERS_PARALLELISM":"false"})
        result=subprocess.run(args,cwd=str(files["runner"].parent),env=env,
                              capture_output=True,text=True,timeout=3600)
        if result.returncode:
            raise MotionBlocked("LTX GPU inference failed: "+result.stderr[-900:])
    clips=list(output.glob("*.mp4"))
    if not clips:
        raise MotionBlocked("Model produced no verifiable MP4")
    return {"state":"GPU_OUTPUT_UNREVIEWED","generated":True,
            "source_sha256":plan["source_sha256"],"files":[{"name":p.name,"sha256":sha(p)}
            for p in clips],"human_review":"REQUIRED","published":False}


def main(argv=None):
    cli=argparse.ArgumentParser(description="Private LTX 2B model adapter, no paid API")
    cli.add_argument("--ltx-repo",type=Path,required=True)
    cli.add_argument("--weights",type=Path,required=True)
    cli.add_argument("--image",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--prompt",required=True)
    cli.add_argument("--execute",action="store_true",help="Explicit CUDA opt-in")
    args=cli.parse_args(argv)
    planned=prepare(args.ltx_repo,args.weights,args.image,args.output,args.prompt)
    result=planned if planned["missing"] else execute(planned,run_gpu=args.execute)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
