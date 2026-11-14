"""Audit actual pinned upstream LTX-Video source compatibility OFFLINE on GitHub.

Extracts exact patch anchor strings from studio's real production worker AST.
Applies the changes to a local checkout of the officially pinned upstream repo,
compiles both patched files, verifies required stage boundaries and model config.
No Kaggle secrets, original actor media, model weights or GPU required.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
from pathlib import Path

WORKER = Path("tools/kaggle_eb01_ltx2b_s02_fallback.py")
PINNED_SHA = "50f0aef1297574a4b29acbdb874e3a363a6d5f2e"


def _get_values(worker: Path) -> dict:
    module=ast.parse(worker.read_text(encoding="utf-8"))
    main=next(n for n in module.body
              if isinstance(n,ast.FunctionDef) and n.name=="main")
    wanted={"old","original_call","upstream_events","updates",
            "stage_events","stage_after"}
    values={}
    for n in ast.walk(main):
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            name=n.targets[0].id
            if name in wanted:
                values[name]=ast.literal_eval(n.value)
    if set(values)!=wanted:
        raise ValueError("Source-pinned production worker patch changed or incomplete")
    return values


def _replace_once(src: str, old: str, new: str, label: str) -> str:
    if src.count(old)!=1:
        raise ValueError(f"PINNED_UPSTREAM_INCOMPATIBLE {label} occurrences={src.count(old)}")
    return src.replace(old,new)


def audit_official_patch(ltx_root: Path,worker: Path = WORKER) -> dict:
    if PINNED_SHA not in worker.read_text(encoding="utf-8"):
        raise ValueError("LTX official git SHA changed without a new audit")
    py=ltx_root/"ltx_video"/"inference.py"
    pipeline=ltx_root/"ltx_video"/"pipelines"/"pipeline_ltx_video.py"
    config=ltx_root/"configs"/"ltxv-2b-0.9.6-distilled.yaml"
    source=py.read_text(encoding="utf-8")
    pipe=pipeline.read_text(encoding="utf-8")
    parameters=_get_values(worker)
    for i,anchor in enumerate(parameters["old"]):
        source=_replace_once(source,anchor,anchor.replace("torch.bfloat16","torch.float16"),
            f"T4_dtype_cast_{i}")
    original=parameters["original_call"]
    source=_replace_once(source,original,
        "        sampler=sampler,\n        device='cpu',\n        enhance_prompt=enhance_prompt,",
        "staged_CPU_loading")
    for i,(old,new) in enumerate(parameters["upstream_events"]):
        source=_replace_once(source,old,new,f"inference_stage_{i}")
    for i,(old,new) in enumerate(parameters["updates"]):
        pipe=_replace_once(pipe,old,new,f"model_device_stage_{i}")
    for i,(old,new) in enumerate(parameters["stage_events"]):
        pipe=_replace_once(pipe,old,new,f"sampler_stage_{i}")
    after=parameters["stage_after"]
    replacement=(
        "        if offload_to_cpu:\n"
        "            self.vae = self.vae.cpu()\n"
        "            torch.cuda.empty_cache()\n"
        "        print('EB01_STAGE_END_CONDITIONING_VAE',flush=True)\n"
        "        self.transformer = self.transformer.to(device)\n"
        "        print('EB01_STAGE_BEGIN_TRANSFORMER',flush=True)\n\n"
        +after
    )
    pipe=_replace_once(pipe,after,replacement,"VAE_then_transformer_only")
    compile(source,str(py),"exec")
    compile(pipe,str(pipeline),"exec")
    markers=(
        "BEGIN_MODEL_LOAD","END_MODEL_LOAD","BEGIN_PROMPT_ENCODE",
        "END_PROMPT_ENCODE","BEGIN_CONDITIONING_VAE",
        "END_CONDITIONING_VAE","BEGIN_TRANSFORMER",
        "DENOISE_STEP_START","DENOISE_STEP_END","END_TRANSFORMER",
        "BEGIN_DECODE","END_DECODE","NATIVE_VIDEO_OUTPUT_READY",
    )
    for marker in markers:
        if (source+pipe).count("EB01_STAGE_"+marker) != 1:
            raise ValueError(f"Missing or ambiguous real GPU stage {marker}")
    cfg=config.read_text(encoding="utf-8")
    if cfg.count("prompt_enhancement_words_threshold: 120")!=1:
        raise ValueError("Official 3B prompt enhancement still unknown")
    if cfg.count("num_inference_steps: 8")!=1:
        raise ValueError("Official native 8-step distilled scheduler changed")
    if cfg.count("checkpoint_path:")!=1:
        raise ValueError("LTX native model checkpoint changed")
    return {
        "schema":1,
        "official_source_revision":PINNED_SHA,
        "upstream_inference_sha256":hashlib.sha256(py.read_bytes()).hexdigest(),
        "upstream_pipeline_sha256":hashlib.sha256(pipeline.read_bytes()).hexdigest(),
        "patched_inference_sha256":hashlib.sha256(source.encode()).hexdigest(),
        "patched_pipeline_sha256":hashlib.sha256(pipe.encode()).hexdigest(),
        "actual_native_scheduler_steps":8,
        "native_stage_labels_compiled":len(markers),
        "python_syntax_and_pinned_patch":"PASS",
        "real_cuda_inference_tested":False,
        "complete_movie_ready":False,
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--ltx-root",type=Path,required=True)
    args=p.parse_args()
    result=audit_official_patch(args.ltx_root)
    for k,v in result.items():
        print(f"EB01_OFFICIAL_LTX_GPU_STATIC_AUDIT {k}={v}")

if __name__=="__main__":
    main()
