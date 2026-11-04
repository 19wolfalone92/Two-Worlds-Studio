"""Real, local LTX image-to-video; separates CPU staging from CUDA inference.

Only the original 2B distilled checkpoint is selected, never the 13B model.
Text embeddings are prepared on CPU before the GPU pass, so T5 XXL never has
to share a 16GB T4 with the video transformer. Artistic approval stays manual.
"""
from __future__ import annotations
import gc
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

REPO = 'Lightricks/LTX-Video'
CHECKPOINT = 'ltxv-2b-0.9.6-distilled-04-25.safetensors'
VAE_REPO = 'Lightricks/LTX-Video-0.9.5'
# The primary implementation's linear-quadratic 8-step distilled schedule.
SIGMAS = [1.0,.99375,.9875,.98125,.975,.909375,.725,.421875]


def distilled_scheduler(base):
    """Keep the trained schedule when an I2V pipeline supplies uniform sigmas.

    Current Diffusers computes uniform sigmas unconditionally, so passing its
    timesteps argument as well raises before the first actual denoising step.
    A scheduler specialization preserves the original distilled 8-step method.
    """
    class OriginalDistilledScheduler(base):
        def set_timesteps(self,num_inference_steps=None,device=None,sigmas=None,mu=None,**kwargs):
            if num_inference_steps not in (None,len(SIGMAS)):
                raise ValueError('This original distilled checkpoint requires eight steps')
            return super().set_timesteps(sigmas=SIGMAS,device=device)
    return OriginalDistilledScheduler


def _digest(path: Path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _json(path: Path, data: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    temp.replace(path)


def stage_weights(dest: Path) -> dict:
    """Download each authorized public shard once and retain compact FP16 files.

    Large original FP32 shards live in transient scratch, not persisted output.
    They are converted one at a time; duplicate raw checkpoints are removed.
    """
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    from diffusers import LTXVideoTransformer3DModel
    dest = Path(dest)
    dest.mkdir(parents=True,exist_ok=True)
    receipt = {'model':REPO,'checkpoint':CHECKPOINT,'gpu_used':False,
               'precision_on_disk':'float16','paid_services':False,'published':False,'sources':[]}
    api = HfApi()
    with tempfile.TemporaryDirectory(prefix='tw-public-ltx-') as temp:
        tmp = Path(temp)
        def component(repo, subfolder):
            info = api.model_info(repo,files_metadata=True)
            files = [f for f in info.siblings if f.rfilename.startswith(subfolder+'/')]
            safe = [f for f in files if f.rfilename.endswith('.safetensors')]
            if not safe:
                raise RuntimeError('Public component has no reviewed safetensors shards: '+subfolder)
            if sum(f.size or 0 for f in safe)>22*1024**3:
                raise RuntimeError('Unexpectedly large public model component')
            selected = safe + [f for f in files if f.rfilename.endswith('.json')]
            for f in selected:
                target = dest/f.rfilename
                if target.is_file():
                    continue
                raw = Path(hf_hub_download(repo,f.rfilename,revision=info.sha,local_dir=tmp/'downloads'))
                target.parent.mkdir(parents=True,exist_ok=True)
                if raw.suffix=='.safetensors':
                    tensors = {}
                    with safe_open(raw,framework='pt',device='cpu') as sf:
                        for key in sf.keys():
                            value = sf.get_tensor(key)
                            if value.is_floating_point():
                                value = value.to(torch.float16)
                                if not torch.isfinite(value).all():
                                    raise RuntimeError('Weight cannot safely be represented in FP16')
                            tensors[key] = value.contiguous()
                    save_file(tensors,str(target),metadata={'format':'pt','source_revision':info.sha})
                    del tensors,value
                    gc.collect()
                else:
                    shutil.copy2(raw,target)
                raw.unlink()
            receipt['sources'].append({'repo':repo,'revision':info.sha,'component':subfolder})
        component(REPO,'text_encoder')
        info = api.model_info(REPO,files_metadata=True)
        raw = Path(hf_hub_download(REPO,CHECKPOINT,revision=info.sha,local_dir=tmp/'checkpoint'))
        if raw.stat().st_size>7*1024**3:
            raise RuntimeError('Refuse a checkpoint larger than the selected original 2B model')
        receipt['original_checkpoint_sha256'] = _digest(raw)
        with safe_open(raw,framework='pt',device='cpu') as sf:
            metadata=sf.metadata() or {}
        if 'config' in metadata:
            _json(dest/'original_checkpoint_config.json',json.loads(metadata['config']))
        transformer = LTXVideoTransformer3DModel.from_single_file(str(raw),
            config=VAE_REPO,subfolder='transformer',torch_dtype=torch.float16)
        if sum(p.numel() for p in transformer.parameters())>3_000_000_000:
            raise RuntimeError('Unexpected transformer size; this worker is explicitly limited to 2B')
        transformer.save_pretrained(dest/'transformer',safe_serialization=True)
        receipt['transformer_parameters']=sum(p.numel() for p in transformer.parameters())
        del transformer
        gc.collect()
        raw.unlink()
        component(VAE_REPO,'vae')
        tokenizer_files=[f for f in api.list_repo_files(REPO,revision=info.sha) if f.startswith('tokenizer/')]
        for name in tokenizer_files:
            raw=Path(hf_hub_download(REPO,name,revision=info.sha,local_dir=tmp/'tokenizer'))
            target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(raw,target)
        raw=Path(hf_hub_download(REPO,'LTX-Video-Open-Weights-License-0.X.txt',revision=info.sha,local_dir=tmp/'license'))
        shutil.copy2(raw,dest/'LTX-Video-Open-Weights-License-0.X.txt')
        receipt['sources'].append({'repo':REPO,'revision':info.sha,'component':CHECKPOINT})
    receipt['staged_bytes']=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())
    if receipt['staged_bytes']>18*1024**3:
        raise RuntimeError('Persistent model output would exceed the production cache budget')
    _json(dest/'stage_receipt.json',receipt)
    return receipt


def encode_directions(source: Path, model_dir: Path, output: Path) -> dict:
    """One CPU T5 load, all private scene directions, no GPU time consumed."""
    import torch
    from transformers import T5EncoderModel,T5Tokenizer
    from safetensors.torch import save_file
    torch.set_num_threads(min(8,torch.get_num_threads()))
    tokenizer=T5Tokenizer.from_pretrained(model_dir/'tokenizer',local_files_only=True)
    encoder=T5EncoderModel.from_pretrained(model_dir/'text_encoder',torch_dtype=torch.float32,
                                         low_cpu_mem_usage=True,local_files_only=True).eval()
    output.mkdir(parents=True,exist_ok=True)
    count=0
    for story in range(13,19):
        path=source/'production_queue'/str(story)/'motion_direction.json'
        directions=json.loads(path.read_text())
        for scene,item in directions['scenes'].items():
            target=output/f'{story}_{int(scene):02d}.safetensors'
            if target.exists():
                count+=1;continue
            values=tokenizer(item['prompt'],padding='max_length',max_length=256,truncation=True,return_tensors='pt')
            with torch.inference_mode():
                embeddings=encoder(input_ids=values.input_ids,attention_mask=values.attention_mask).last_hidden_state
            if not torch.isfinite(embeddings).all():
                raise RuntimeError('Non-finite CPU text embeddings')
            save_file({'prompt_embeds':embeddings.contiguous(),
                       'prompt_attention_mask':values.attention_mask.contiguous()},str(target),
                      metadata={'source_sha256':hashlib.sha256(item['prompt'].encode()).hexdigest()})
            count+=1
            _json(output/'embedding_status.json',{'scenes':count,'gpu_used':False})
    del encoder
    gc.collect()
    return {'scenes':count,'gpu_used':False}


def load_pipeline(model_dir: Path):
    import torch
    from diffusers import LTXImageToVideoPipeline,LTXVideoTransformer3DModel,AutoencoderKLLTXVideo,FlowMatchEulerDiscreteScheduler
    if not torch.cuda.is_available():
        raise RuntimeError('A real CUDA device is required for character video generation')
    # FP32 avoids BF16-only kernels and FP16 overflows on the actual T4.
    transformer=LTXVideoTransformer3DModel.from_pretrained(model_dir/'transformer',
                    torch_dtype=torch.float32,local_files_only=True)
    vae=AutoencoderKLLTXVideo.from_pretrained(model_dir/'vae',torch_dtype=torch.float32,local_files_only=True)
    scheduler=distilled_scheduler(FlowMatchEulerDiscreteScheduler)(num_train_timesteps=1000,shift=1.0,
                    use_dynamic_shifting=False,stochastic_sampling=True)
    pipe=LTXImageToVideoPipeline(transformer=transformer,vae=vae,scheduler=scheduler,
                                text_encoder=None,tokenizer=None)
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()
    return pipe


def render_clip(pipe, embeddings: Path, image: Path, dest: Path, *, seed: int,
                width=512,height=896,frames=121,fps=24) -> dict:
    import torch
    import numpy as np
    from PIL import Image,ImageOps
    from safetensors.torch import load_file
    from diffusers.utils import export_to_video
    if dest.exists() or width%32 or height%32 or (frames-1)%8 or not 49<=frames<=257:
        raise ValueError('Existing destination or invalid bounded LTX dimensions')
    values=load_file(str(embeddings))
    # The precomputed CPU embeddings bypass the text encoder. Diffusers keeps
    # their original device, while model offload moves the transformer to T4.
    values={name:tensor.to('cuda:0') for name,tensor in values.items()}
    with Image.open(image) as im:
        reference=ImageOps.fit(im.convert('RGB'),(width,height),method=Image.Resampling.LANCZOS)
    start=time.monotonic()
    result=pipe(image=reference,prompt_embeds=values['prompt_embeds'],
                prompt_attention_mask=values['prompt_attention_mask'],width=width,height=height,
                num_frames=frames,frame_rate=fps,guidance_scale=1.0,
                num_inference_steps=len(SIGMAS),
                decode_timestep=.05,decode_noise_scale=.025,
                generator=torch.Generator(device='cuda:0').manual_seed(seed)).frames[0]
    if len(result)!=frames:
        raise RuntimeError('Neural model returned the wrong number of actual frames')
    samples=[np.asarray(result[i],dtype=np.float32)/255 for i in [0,len(result)//2,len(result)-1]]
    if any(not np.isfinite(a).all() or a.std()<.01 for a in samples):
        raise RuntimeError('Neural model returned blank or non-finite video')
    dest.parent.mkdir(parents=True,exist_ok=True)
    export_to_video(result,str(dest),fps=fps,macro_block_size=1,quality=8)
    receipt={'model':REPO,'checkpoint':CHECKPOINT,'actual_neural_frames':frames,
             'seconds':frames/fps,'width':width,'height':height,'seed':seed,
             'gpu':torch.cuda.get_device_name(0),'gpu_memory_pooled':False,
             'compute_dtype':'float32','elapsed_s':round(time.monotonic()-start,2),
             'inference_sigmas':SIGMAS,
             'gpu_peak_allocated_gib':round(torch.cuda.max_memory_allocated()/1024**3,3),
             'sha256':_digest(dest),
             'character_motion_review':'pending','artistic_review':'pending','published':False}
    _json(dest.with_suffix('.receipt.json'),receipt)
    return receipt
