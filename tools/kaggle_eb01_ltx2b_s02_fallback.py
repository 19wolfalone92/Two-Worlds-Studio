"""Genuine alternative native I2V for corrupted S02. Private Kaggle GPU only.
Uses source-locked original portrait, official LTX 2B, transparent T4-FP16 patch.
No paid model, no mock video, no auto-publish, no false movie claims.
"""
from __future__ import annotations
import base64,hashlib,json,os,re,shutil,subprocess,sys,traceback
from pathlib import Path

CONFIG_B64="__EB01_LTX2B_CONFIG_B64__"
ROOT=Path("/kaggle/temp/eb01-ltx-studio")
LTX=Path("/kaggle/temp/eb01-LTX-Video")
INPUT=Path("/kaggle/temp/eb01-ltx-locked-input")
OUT=Path("/kaggle/working")
JOB=OUT/"EB01_S02_LTX_PRIVATE_JOB.json"
LTX_SHA="50f0aef1297574a4b29acbdb874e3a363a6d5f2e"

def run(args,timeout=360,cwd=None):
    p=subprocess.run([str(x) for x in args],cwd=cwd,text=True,
                     capture_output=True,timeout=timeout)
    if p.returncode:
        print("EB01_COMMAND_BLOCKED",p.stderr[-1400:],p.stdout[-1100:],flush=True)
        raise RuntimeError("Subprocess failed "+str(args[:3]))
    return p.stdout

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for part in iter(lambda:f.read(2*1024*1024),b""):
            h.update(part)
    return h.hexdigest()

def main():
    cfg=json.loads(base64.b64decode(CONFIG_B64,validate=True))
    if not (cfg.get("schema")==7 and cfg.get("shot")=="S02"
        and cfg.get("private") is True and cfg.get("no_paid_services") is True
        and cfg.get("new_gpu_jobs")==1 and cfg.get("allow_social_publish") is False
        and cfg.get("source_dataset")=="wolf19loner92/eb01-v7-private-production-sources"
        and cfg.get("branch")=="fix/eb01-s02-private-raw-preservation-20261113"
        and re.fullmatch(r"[a-f0-9]{40}",str(cfg.get("revision","")))):
        raise RuntimeError("Unsafe/changed original private I2V request")
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<14*2**30:
            raise RuntimeError("Real GPU >=14 GiB unavailable; no CPU fake")
        print("EB01_ALT_ENGINE_REAL_CUDA",torch.cuda.get_device_name(0),
              "bf16_native",torch.cuda.is_bf16_supported(),flush=True)
        run(["git","clone","--depth","1","--branch",cfg["branch"],
             "https://github.com/19wolfalone92/Two-Worlds-Studio.git",ROOT],timeout=240)
        run(["git","fetch","--depth","1","origin",cfg["revision"]],cwd=ROOT)
        run(["git","checkout","--detach",cfg["revision"]],cwd=ROOT)
        if run(["git","rev-parse","HEAD"],cwd=ROOT).strip()!=cfg["revision"]:
            raise RuntimeError("Original studio revision mismatch")
        sys.path.insert(0,str(ROOT))
        from studio.eb01_v7_storyboard import gather_private,SRC_SHA,VOICE_SHA,SHOTS,sha as source_sha
        from tools.kaggle_eb01_v7_full_entry import REVIEWED_SOURCE_SHA
        lock=gather_private(Path("/kaggle/input"),INPUT)
        if lock["source_sha256"]!=SRC_SHA or lock["voice_sha256"]!=VOICE_SHA or (
             {k:r["reference_sha256"] for k,r in lock["scenes"].items()} != REVIEWED_SOURCE_SHA):
            raise RuntimeError("Actual original image/six Russian voice SHA contract broken")
        reference=INPUT/"frames"/"S02.png"
        scene=next(x for x in SHOTS if x[0]=="S02")
        if scene[2]!=125 or source_sha(reference)!=REVIEWED_SOURCE_SHA["S02"]:
            raise RuntimeError("Original Kostya still or 125-frame storyboard changed")
        print("EB01_ALT_ENGINE_ORIGINAL_PHOTO_AND_VOICE_LOCK_PASS",flush=True)
        run(["git","clone","--depth","1","https://github.com/Lightricks/LTX-Video.git",LTX],timeout=240)
        run(["git","fetch","--depth","1","origin",LTX_SHA],cwd=LTX)
        run(["git","checkout","--detach",LTX_SHA],cwd=LTX)
        if run(["git","rev-parse","HEAD"],cwd=LTX).strip()!=LTX_SHA:
            raise RuntimeError("Official LTX model implementation not pinned")
        # NVIDIA T4 is pre-Ampere (no native BF16). Document exactly three
        # compatibility casts, retaining true LTX neural inference and weights.
        source=LTX/"ltx_video"/"inference.py"
        code=source.read_text()
        old=[
          "return Transformer3DModel.from_pretrained(ckpt_path).to(torch.bfloat16)",
          "vae = vae.to(torch.bfloat16)",
          "text_encoder = text_encoder.to(torch.bfloat16)"
        ]
        for item in old:
            if code.count(item)!=1:raise RuntimeError("Official LTX source changed; cannot safely patch T4")
            code=code.replace(item,item.replace("torch.bfloat16","torch.float16"))
        # ROOT CAUSE VERIFIED IN FIRST LTX GPU LOG: ~14.42/14.56 GiB
        # exhausted while create_ltx_video_pipeline simultaneously places
        # transformer, T5 and VAE on CUDA, *before* offload_to_cpu can run.
        # Stage all original official weights on CPU. Inside the sampler,
        # move T5 -> CPU, transformer -> CPU, VAE -> CUDA sequentially.
        # Keep the full actual LTX inference on the real T4, no fake frames.
        original_call="        sampler=sampler,\n        device=device,\n        enhance_prompt=enhance_prompt,"
        if code.count(original_call)!=1:
            raise RuntimeError("Official LTX pipeline call changed; memory fix cannot be verified")
        code=code.replace(original_call,
            "        sampler=sampler,\n        device='cpu',\n        enhance_prompt=enhance_prompt,")
        source.write_text(code)
        stages=LTX/"ltx_video"/"pipelines"/"pipeline_ltx_video.py"
        pipe=stages.read_text()
        updates=[
          ("        device = self._execution_device\n\n        self.video_scale_factor",
           "        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n\n        self.video_scale_factor"),
          ("self.text_encoder = self.text_encoder.to(self._execution_device)",
           "self.text_encoder = self.text_encoder.to(device)"),
          ("self.transformer = self.transformer.to(self._execution_device)",
           "self.transformer = self.transformer.to(device)"),
          (").to(dtype=init_latents.dtype)\n",
           ").to(dtype=init_latents.dtype, device=init_latents.device)\n"),
          ("            image = vae_decode(\n",
           "            self.vae = self.vae.to(device)\n            image = vae_decode(\n"),
        ]
        for previous,updated in updates:
            if pipe.count(previous)!=1:
                raise RuntimeError("Official LTX pipeline stage changed; unsafe GPU memory patch")
            pipe=pipe.replace(previous,updated)
        stages.write_text(pipe)
        patched=sha(source)
        stage_sha=sha(stages)
        print("EB01_ALT_ENGINE_ROOT_CAUSE_OOM_FIXED_SEQUENTIAL_CPU_GPU_STAGE",
              "inference_sha",patched,"pipeline_sha",stage_sha,flush=True)
        run([sys.executable,"-m","pip","install","-q","-e",str(LTX)+"[inference]"],timeout=960)
        c=LTX/"configs"/"ltxv-2b-0.9.6-distilled.yaml"
        if not c.is_file() or "ltxv-2b-0.9.6-distilled-04-25.safetensors" not in c.read_text():
            raise RuntimeError("Expected authentic LTX 2B model not configured")
        generated=OUT/"EB01_S02_LTX_REAL_GENERATION_PRIVATE"
        generated.mkdir(exist_ok=False)
        script="""from ltx_video.inference import infer,InferenceConfig
import os
settings=InferenceConfig(pipeline_config=os.environ['LTX_CONFIG'],prompt=os.environ['LTX_PROMPT'],
  seed=202611132,height=576,width=320,num_frames=125,frame_rate=24,
  offload_to_cpu=True,conditioning_media_paths=[os.environ['LTX_INPUT']],
  conditioning_start_frames=[0],image_cond_noise_scale=0.025,
  output_path=os.environ['LTX_OUTPUT'])
print('EB01_LTX_REAL_NATIVE_125_FRAME_I2V_INFERENCE_START',flush=True)
infer(settings)
"""
        env=os.environ.copy()
        env.update(LTX_CONFIG=str(c),LTX_PROMPT=scene[-1],LTX_INPUT=str(reference),
                   LTX_OUTPUT=str(generated))
        proc=subprocess.run([sys.executable,"-c",script],cwd=LTX,env=env,
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=2600)
        (OUT/"EB01_LTX_PRIVATE_GPU_LOG.txt").write_text(proc.stdout[-40000:])
        if proc.returncode:
            print("EB01_ACTUAL_LTX_INFERENCE_FAILURE",proc.stdout[-4500:],flush=True)
            raise RuntimeError("Native LTX 2B inference process failed exit "+str(proc.returncode))
        clips=list(generated.rglob("*.mp4"))
        if len(clips)!=1:raise RuntimeError("No unique actual generated video "+str(len(clips)))
        video=OUT/"S02.mp4"
        shutil.copy2(clips[0],video)
        from studio.comfy_neural import _probe
        from studio.eb01_motion_artifact_gate import inspect_clip
        meta=_probe(video)
        qc=inspect_clip(video)
        run(["ffmpeg","-v","error","-xerror","-i",video,"-f","null","-"],timeout=180)
        print("EB01_ALT_REAL_GPU_ACTUAL_MODEL_FRAMES",meta["frames"],
              "artifact_gate",qc["status"],flush=True)
        if (meta["frames"]!=125 or meta["fps"]!="24" or
            meta["width"]!=320 or meta["height"]!=576 or
            qc["decoded_frames"]!=125 or qc["status"]!="NO_NEON_ARTIFACT_FOUND"):
            raise RuntimeError("Authentic LTX S02 failed required video frame/color QA")
        proof={"schema":7,"shot":"S02","success":True,
           "status":"ONE_REAL_LTX2B_S02_TECHNICALLY_VERIFIED_OWNER_REVIEW_REQUIRED",
           "model":"Lightricks/LTX-Video-2B-0.9.6-distilled",
           "engine":"ltx_official_fp16_t4_compatibility",
           "official_source_git_sha":LTX_SHA,"compatibility_patch_sha256":patched,
           "pipeline_memory_patch_sha256":stage_sha,
           "studio_git_revision":cfg["revision"],
           "original_reference_sha256":source_sha(reference),
           "original_character_sha256":SRC_SHA["KOSTYA"],
           "six_original_russian_voice_sha256":VOICE_SHA,
           "clip_sha256":source_sha(video),"frame_count":125,"fps":24,
           "width":320,"height":576,"true_gpu_neural_inference":True,
           "invideo_used":False,"paid_inference_used":False,
           "human_artistic_approved":False,"whole_movie_ready":False,"published":False}
        JOB.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n")
        print("EB01_ACTUAL_LTX_S02_FULL_NATIVE_VIDEO_PRODUCED",proof["clip_sha256"],flush=True)
    except BaseException as exc:
        JOB.write_text(json.dumps({"schema":7,"shot":"S02",
          "status":"REAL_ALTERNATIVE_GPU_BLOCKED_NO_FALSE_FILM_CLAIM",
          "success":False,"error_type":type(exc).__name__,
          "error_summary":str(exc)[:800],"published":False},ensure_ascii=False,indent=2)+"\n")
        print("EB01_ALT_MODEL_INFERENCE_BLOCKED",type(exc).__name__,str(exc)[:700],flush=True)
        traceback.print_exc()
        raise

if __name__=="__main__":
    main()
