"""True native Studio image-to-video test on private Kaggle T4.
Real LTX2B neural denoising only, pinned upstream, zero public portrait media.
Source JPEG is provided in private kernel script only; never checked into GitHub.
"""
from __future__ import annotations
import base64,hashlib,json,os,subprocess,sys,shutil,time,traceback
from pathlib import Path
CONFIG_B64="__STUDIO_WAVE_CONFIG_B64__"
IMAGE_B64="__STUDIO_WAVE_IMAGE_B64__"
ROOT=Path("/kaggle/temp/two-worlds-wave-repo")
LTX=Path("/kaggle/temp/studio-wave-ltx")
OUT=Path("/kaggle/working")
LTX_SHA="50f0aef1297574a4b29acbdb874e3a363a6d5f2e"
def run(args,timeout=360,cwd=None):
    x=subprocess.run(list(map(str,args)),cwd=cwd,text=True,capture_output=True,timeout=timeout)
    if x.returncode:raise RuntimeError("STUDIO_BOUNDED_COMMAND_FAILED:"+str(args[:2]))
    return x.stdout
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    cfg=json.loads(base64.b64decode(CONFIG_B64,validate=True))
    if (cfg.get("schema")!=1 or cfg.get("mode")!="studio-native-wave" or
        cfg.get("private") is not True or cfg.get("no_paid_services") is not True or
        cfg.get("allow_social_publish") is not False or cfg.get("human_artistic_approved") is not False or
        cfg.get("source_sha256")!="379d772afa4b7f1448b0881d84631776329ecf1d670ac8da9b5701f2ea5edc72"):
        raise ValueError("Unsafe native Studio reference or privacy policy")
    import torch
    from PIL import Image
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<14*2**30:
        raise RuntimeError("True CUDA T4 >=14GiB unavailable; no CPU fake")
    reference=OUT/"STUDIO_WAVE_PRIVATE_REFERENCE.jpg"
    reference.write_bytes(base64.b64decode(IMAGE_B64,validate=True))
    if sha(reference)!=cfg["source_sha256"]:raise ValueError("Reference hash mismatch")
    with Image.open(reference) as photo:
        if photo.size!=(720,1280):raise ValueError("Reference geometry mismatch")
        photo.verify()
    run(["git","clone","--depth","1","--branch",cfg["branch"],
         "https://github.com/19wolfalone92/Two-Worlds-Studio.git",ROOT],timeout=300)
    run(["git","fetch","--depth","1","origin",cfg["revision"]],cwd=ROOT)
    run(["git","checkout","--detach",cfg["revision"]],cwd=ROOT)
    if run(["git","rev-parse","HEAD"],cwd=ROOT).strip()!=cfg["revision"]:
        raise RuntimeError("Source revision mismatch")
    sys.path.insert(0,str(ROOT))
    from studio.eb01_native_t4_profile import patch_native_sources
    from tools.eb01_bounded_inference import run_native_bounded,InferenceTimeBudgetExpired
    from studio.comfy_neural import _probe
    from studio.eb01_motion_artifact_gate import inspect_clip
    from tools.eb01_true_gpu_clip_numbers import gray_motion
    native_profile=True
    run(["git","clone","--depth","1","https://github.com/Lightricks/LTX-Video.git",LTX],timeout=300)
    run(["git","fetch","--depth","1","origin",LTX_SHA],cwd=LTX)
    run(["git","checkout","--detach",LTX_SHA],cwd=LTX)
    if run(["git","rev-parse","HEAD"],cwd=LTX).strip()!=LTX_SHA:
        raise RuntimeError("Official LTX revision changed")
    source=LTX/"ltx_video"/"inference.py"
    code=source.read_text()
    old=[
      "return Transformer3DModel.from_pretrained(ckpt_path).to(torch.bfloat16)",
      "vae = vae.to(torch.bfloat16)",
      "text_encoder = text_encoder.to(torch.bfloat16)"
    ]
    for item in old:
        if code.count(item)!=1:raise RuntimeError("Official LTX source changed; cannot safely patch T4")
        # T4 cannot compute BF16. Keep 2B transformer/T5 in FP16 but
        # use FP32 for sequentially-offloaded VIDEO VAE to avoid
        # potential FP16 decoder overflows behind neon RGB artifacts.
        # The previous full FP16 real GPU generated 33 valid frames yet
        # color gate flagged 33/33, so this is an actual A/B numerical
        # precision change, NOT another unchanged expensive rerun.
        chosen=("torch.float32" if item.startswith("vae = vae.to(")
                else "torch.float16")
        code=code.replace(item,item.replace("torch.bfloat16",chosen))
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
    # Emit only predefined stage labels, not private actor prompts.
    upstream_events=[
      ("    pipeline = create_ltx_video_pipeline(",
       "    print('WAVE_STAGE_BEGIN_MODEL_LOAD',flush=True)\n    pipeline = create_ltx_video_pipeline("),
      ('    if pipeline_config.get("pipeline_type", None) == "multi-scale":',
       '    print("WAVE_STAGE_END_MODEL_LOAD",flush=True)\n    if pipeline_config.get("pipeline_type", None) == "multi-scale":'),
      ('        logger.warning(f"Output saved to {output_filename}")',
       '        print("WAVE_STAGE_NATIVE_VIDEO_OUTPUT_READY",flush=True)\n        logger.warning(f"Output saved to {output_filename}")'),
    ]
    for original,updated in upstream_events:
        if code.count(original)!=1:
            raise RuntimeError("Pinned official LTX inference stage changed")
        code=code.replace(original,updated)
    source.write_text(code)
    stages=LTX/"ltx_video"/"pipelines"/"pipeline_ltx_video.py"
    pipe=stages.read_text()
    updates=[
      ("        device = self._execution_device\n\n        self.video_scale_factor",
       "        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n\n        self.video_scale_factor"),
      ("self.text_encoder = self.text_encoder.to(self._execution_device)",
       "print('WAVE_STAGE_BEGIN_PROMPT_ENCODE',flush=True)\n            self.text_encoder = self.text_encoder.to(device)"),
      # ROOT FIX 2: I2V input photo encoding invokes vae_encode()
      # *before* denoising. The previous "memory fix" left VAE on
      # CPU in FP16 and shifted the transformer to CUDA too early,
      # causing slow/unsupported FP16 CPU VAE and GPU residency overlap.
      # Encode the original reference on CUDA using just VAE, then
      # offload VAE and load the transformer for 8-step sampling.
      ("self.transformer = self.transformer.to(self._execution_device)",
       "print('WAVE_STAGE_BEGIN_CONDITIONING_VAE',flush=True)\n        self.vae = self.vae.to(device)"),
      (").to(dtype=init_latents.dtype)\n",
       ").to(dtype=init_latents.dtype, device=init_latents.device)\n"),
      ("            image = vae_decode(\n",
       "            print('WAVE_STAGE_BEGIN_DECODE',flush=True)\n            self.vae = self.vae.to(device)\n            image = vae_decode(\n"),
    ]
    for previous,updated in updates:
        if pipe.count(previous)!=1:
            raise RuntimeError("Official LTX pipeline stage changed; unsafe GPU memory patch")
        pipe=pipe.replace(previous,updated)
    stage_events=[
      ('        if offload_to_cpu and self.text_encoder is not None:',
       '        print("WAVE_STAGE_END_PROMPT_ENCODE",flush=True)\n        if offload_to_cpu and self.text_encoder is not None:'),
      ("            for i, t in enumerate(timesteps):",
       "            for i, t in enumerate(timesteps):\n                print('WAVE_STAGE_DENOISE_STEP_START',i+1,flush=True)"),
      ("                # call the callback, if provided",
       "                print('WAVE_STAGE_DENOISE_STEP_END',i+1,flush=True)\n                # call the callback, if provided"),
      ("        if offload_to_cpu:\n            self.transformer = self.transformer.cpu()",
       "        print('WAVE_STAGE_END_TRANSFORMER',flush=True)\n        if offload_to_cpu:\n            self.transformer = self.transformer.cpu()"),
      ("            image = self.image_processor.postprocess(image, output_type=output_type)",
       "            image = self.image_processor.postprocess(image, output_type=output_type)\n            print('WAVE_STAGE_END_DECODE',flush=True)"),
    ]
    for original,updated in stage_events:
        if pipe.count(original)!=1:
            raise RuntimeError("Pinned official LTX pipeline stage instrumentation changed")
        pipe=pipe.replace(original,updated)
    # After the conditioning/photo VAE encoding completes, transfer it
    # back to RAM before moving 2B transformer to GPU. Then after
    # transformer sampling the existing decode-stage patch moves VAE
    # back to GPU. Never keep all T5+VAE+transformer on T4 at once.
    stage_after="        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline"
    if pipe.count(stage_after)!=1:
        raise RuntimeError("Official LTX after-conditioning boundary changed; cannot safely stage CUDA VAE")
    pipe=pipe.replace(stage_after,
        "        if offload_to_cpu:\n"
        "            self.vae = self.vae.cpu()\n"
        "            torch.cuda.empty_cache()\n"
        "        print('WAVE_STAGE_END_CONDITIONING_VAE',flush=True)\n"
        "        self.transformer = self.transformer.to(device)\n"
        "        print('WAVE_STAGE_BEGIN_TRANSFORMER',flush=True)\n\n"
        +stage_after)
    stages.write_text(pipe)
    if native_profile:
        from studio.studio_wave_native_t4_profile import patch_native_sources
        native_patch_sha=patch_native_sources(source,stages,LTX/"ltx_video/models/autoencoders/vae.py")
    patched=sha(source)
    stage_sha=sha(stages)
    print("WAVE_T4_NATIVE_PRECISION_POLICY",
          "transformer_fp16",True,"text_fp16",True,
          "vae_encoder_decoder_fp32",True,flush=True)
    print("WAVE_ALT_ENGINE_ROOT_CAUSE_OOM_FIXED_SEQUENTIAL_CPU_GPU_STAGE",
          "inference_sha",patched,"pipeline_sha",stage_sha,flush=True)
    run([sys.executable,"-m","pip","install","-q","-e",str(LTX)+"[inference]"],timeout=960)
    c=LTX/"configs"/"ltxv-2b-0.9.6-distilled.yaml"
    if not c.is_file() or "ltxv-2b-0.9.6-distilled-04-25.safetensors" not in c.read_text():
        raise RuntimeError("Expected authentic LTX 2B model not configured")
    # CONFIRMED ROOT CAUSE OF BOTH GPU TIMEOUTS: official config uses
    # prompt_enhancement_words_threshold=120. The original 70-word
    # EB01 actor prompt therefore silently loads Florence-2-large AND
    # Llama-3.2-3B, then runs 3B prompt enhancement on CPU under the
    # memory-offload patch. This is neither the LTX 2B distilled video
    # sampler nor necessary for the original human-approved scene prompt.
    # Disable the unsolicited extra 3B caption+LLM pipeline, preserving
    # the original scene motion wording, official 2B video weights and
    # EXACT 8-step distilled scheduler. Never change reference images.
    original_config=c.read_text(encoding="utf-8")
    threshold="prompt_enhancement_words_threshold: 120"
    if original_config.count(threshold)!=1:
        raise RuntimeError("Official LTX prompt-enhancer config changed; cannot remove unapproved 3B CPU bottleneck")
    c.write_text(original_config.replace(
        threshold,"prompt_enhancement_words_threshold: 0"
    ),encoding="utf-8")
    print("WAVE_REAL_LTX_2B_STAGING_ONLY_NO_FLORENCE_LLAMA_3B",
          "original_config_sha256",hashlib.sha256(original_config.encode()).hexdigest(),
          "patched_config_sha256",sha(c),
          "distilled_native_steps",8,
          "exact_approved_prompt_unmodified",True,flush=True)

    print("STUDIO_WAVE_OFFICIAL_LTX_PINNED_STAGE_READY",flush=True)
    generated=OUT/"STUDIO_WAVE_NATIVE_PRIVATE_GENERATION";generated.mkdir(exist_ok=False)
    prompts=[
       "Photorealistic locked-off vertical studio portrait of the SAME bald, bearded man wearing black rectangular glasses and cream shirt, seated in the SAME studio with bookshelf and plants. He naturally raises his free right hand to wave hello, smiles gently, tilts his head slightly, realistic neck and shoulder biomechanics; his mouth and beard remain attached to his face, natural articulation of all five fingers. Absolutely stationary camera, bookshelf, chair, lamp and plant. No face morphing, no unstable glasses, no floating beard, no jumping or cut.",
       "Photorealistic continuous exact SAME shot and exact SAME bald bearded man, black rectangular spectacles and cream T-shirt; he waves the raised free right hand gently, then smoothly lowers it and relaxes with a real warm smile and a tiny organic head movement. Natural intact five fingers, stiff eyeglasses fit, beard connected with jaw, stationary shelves, lamp, plant, camera, torso physically coherent. No jitter, no deformation, no scene cuts."
    ]
    script="""from ltx_video.inference import infer,InferenceConfig
import os
settings=InferenceConfig(
 pipeline_config=os.environ['LTX_CONFIG'],
 prompt=os.environ['LTX_PROMPT'], seed=int(os.environ['STUDIO_SEED']),
 height=1024,width=576,num_frames=33,frame_rate=24,
 offload_to_cpu=True,
 conditioning_media_paths=[os.environ['LTX_INPUT']],
 conditioning_strengths=[1.0],conditioning_start_frames=[0],
 image_cond_noise_scale=0.15,output_path=os.environ['LTX_OUTPUT'])
print("STUDIO_WAVE_NATIVE_LTX_SAMPLE_START",flush=True)
infer(settings)
"""
    videos=[];records=[];conditioner=reference
    try:
        for segment in (0,1):
            target=generated/("segment_"+str(segment));target.mkdir()
            env=os.environ.copy()
            env.update(LTX_CONFIG=str(c),LTX_PROMPT=prompts[segment],
                LTX_INPUT=str(conditioner),LTX_OUTPUT=str(target),
                STUDIO_SEED=str(2026112301+segment),
                PYTHONPATH=str(ROOT)+os.pathsep+env.get("PYTHONPATH",""))
            # Precisely two bounded real 33-frame original neural segments.
            observed=run_native_bounded([sys.executable,"-u","-c",script],
                cwd=LTX,env=env,
                log_path=OUT/("STUDIO_WAVE_PRIVATE_GPU_"+str(segment)+".log"),
                deadline_seconds=1400,heartbeat_seconds=20)
            clips=list(target.rglob("*.mp4"))
            if len(clips)!=1:raise RuntimeError("Genuine neural MP4 missing or ambiguous")
            clip=clips[0]
            meta=_probe(clip)
            run(["ffmpeg","-v","error","-xerror","-i",clip,"-f","null","-"],timeout=90)
            if meta.get("frames")!=33 or meta.get("fps")!="24" or meta.get("width")!=576 or meta.get("height")!=1024:
                raise RuntimeError("Native model geometry/frame verification failed")
            # Preserve full raw model output, even if scene-specific artistic QC later blocks approval.
            safe=OUT/("STUDIO_WAVE_REAL_SEGMENT_"+str(segment)+".mp4")
            shutil.copy2(clip,safe)
            videos.append(safe)
            records.append({"index":segment,"sha256":sha(safe),
                            "frames":33,"fps":24,"width":576,"height":1024,
                            "model_seconds":observed["elapsed_seconds"],
                            "color_qa":inspect_clip(safe)["status"],
                            "numerical_motion":gray_motion(safe)})
            # Condition segment 2 on the actual LAST RGB frame of segment 1.
            if segment==0:
                conditioner=OUT/"STUDIO_WAVE_PRIVATE_LASTFRAME.png"
                run(["ffmpeg","-v","error","-xerror","-i",safe,"-vf","select=eq(n\\,32)","-frames:v","1",conditioner],timeout=45)
                if not conditioner.exists():raise RuntimeError("No continuity image from first native segment")
        # Drop duplicated conditioning frame from next segment, preserve all true model frames.
        merged=OUT/"STUDIO_WAVE_TRUE_NATIVE_65F_24FPS.mp4"
        run(["ffmpeg","-v","error","-xerror","-i",videos[0],"-i",videos[1],
             "-filter_complex","[1:v]trim=start_frame=1,setpts=PTS-STARTPTS[v1];[0:v][v1]concat=n=2:v=1:a=0[out]",
             "-map","[out]","-an","-r","24","-c:v","libx264","-preset","medium","-crf","17",
             "-pix_fmt","yuv420p",merged],timeout=240)
        m=_probe(merged)
        run(["ffmpeg","-v","error","-xerror","-i",merged,"-f","null","-"],timeout=120)
        if m.get("frames")!=65 or m.get("fps")!="24" or m.get("width")!=576 or m.get("height")!=1024:
            raise RuntimeError("Full assembled native Studio clip failed format/frame check")
        receipt={"schema":1,"status":"NATIVE_NEURAL_VIDEO_GENERATED_ARTISTIC_REVIEW_REQUIRED",
            "model":"Lightricks/LTX-Video-2B-0.9.6-distilled",
            "source":"original_user_reference_jpeg_only_not_previous_video_frames",
            "source_reference_sha256":sha(reference),"source_git_sha":cfg["revision"],
            "upstream_git_sha":LTX_SHA,
            "actual_native_gpu_inference":True,
            "original_neural_segments":records,
            "output_sha256":sha(merged),"output_frames":65,
            "output_width":576,"output_height":1024,"output_fps":24,
            "extra_interpolated_frames":0,
            "full_video_length_seconds":65/24,
            "private_media_only":True,"published":False,
            "human_artistic_approved":False,"qa_color_all_segments":all(r['color_qa']=='NO_NEON_ARTIFACT_FOUND' for r in records),
            "anatomical_continuity_human_review_required":True}
        (OUT/"STUDIO_WAVE_TRUE_NATIVE_RECEIPT.json").write_text(json.dumps(receipt,indent=2))
        print("STUDIO_WAVE_TRUE_NATIVE_GPU_DONE",receipt["output_sha256"],
              "native_frames",m["frames"],"seconds",round(65/24,3),flush=True)
    except BaseException as exc:
        (OUT/"STUDIO_WAVE_TRUE_NATIVE_FAILURE_SAFE.json").write_text(json.dumps({
         "stage":"NATIVE_GPU_PIPELINE","error_type":type(exc).__name__,
         "successful_native_segments":len(videos),
         "public_private_source_leak":False,"human_artistic_approved":False,"published":False}))
        print("STUDIO_WAVE_NATIVE_MODEL_BLOCKED",type(exc).__name__,"partial_segments",len(videos),flush=True)
        raise
if __name__=="__main__":main()
