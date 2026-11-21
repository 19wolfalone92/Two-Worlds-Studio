"""Restore the frozen real S01 pilot after denoising, without rerunning diffusion.

This is recovery evidence, never a new original take or an artistic approval.
Only the original SHA-locked FP16 latents and the official FP32 VAE are used.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,struct,subprocess,sys,time
from pathlib import Path

SOURCE_RUN='44818907329'
SOURCE_REVISION='d134481d50c7d27f7948a6cb82318504189bd672'
LATENT_SHA='6ce62fc8b42352ad9d0398fa87f0568749376a35b000cff39ae8c7e0da45ca6f'
ORIGINAL_MP4_SHA='f0eef4c2a4278e15d0e9eca31837e37ec01c1e00871664090f06f9cc49324577'
VAE_REVISION='fc913c34361f4ec879e2f9c78b4f11ae50a937d1'
VAE_SHA='38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4<<20),b''):h.update(part)
    return h.hexdigest()


def checkpoint_info(path,expected_sha=LATENT_SHA):
    """Inspect without importing CUDA or unpickling an arbitrary checkpoint."""
    path=Path(path)
    if sha(path)!=expected_sha:raise ValueError('Saved denoised checkpoint SHA mismatch')
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        if not 1<=n<=16000:raise ValueError('Invalid checkpoint header length')
        doc=json.loads(f.read(n))
    if doc!={'latents':{'dtype':'F16','shape':[1,16,17,128,72],'data_offsets':[0,5013504]}}:
        raise ValueError('Checkpoint tensor name, shape, dtype or byte offsets changed')
    if path.stat().st_size!=8+n+5013504:raise ValueError('Truncated checkpoint payload')
    return {'checkpoint_sha256':expected_sha,'tensor_shape':[1,16,17,128,72],
        'dtype':'F16','native_decoded_frames':65,'restoration_only':True,'new_denoising_steps':0}


def find_original(mount,name,digest):
    paths=[p for p in Path(mount).rglob(name) if p.is_file() and sha(p)==digest]
    if len(paths)!=1:raise ValueError('Exact frozen original input missing or ambiguous')
    return paths[0]


def restore(config_path):
    import torch
    import numpy as np
    import imageio.v2 as imageio
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from studio.eb01_echo_profile import UPSTREAM_SHA,pilot_spec
    from tools.eb01_resource_sampler import start
    cfg=json.loads(Path(config_path).read_text())
    if (cfg.get('restoration_only') is not True or cfg.get('account_alias')!='collaborator' or
        cfg.get('source_run')!=SOURCE_RUN or cfg.get('upstream_revision')!=UPSTREAM_SHA):
        raise ValueError('Unapproved recovery request')
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA VAE decoder required')
    runtime=Path(cfg['runtime']);work=Path(cfg['work']);out=Path('/kaggle/working');mount=Path('/kaggle/input')
    actual=subprocess.check_output(['git','-C',str(runtime),'rev-parse','HEAD'],text=True).strip()
    if actual!=UPSTREAM_SHA:raise ValueError('Original VAE implementation revision changed')
    saved=find_original(mount,'EB01_ECHO_DENOISED_LATENTS.safetensors',LATENT_SHA)
    proof=checkpoint_info(saved)
    original=find_original(mount,'EB01_ECHO_RUSSIAN_NATIVE.mp4',ORIGINAL_MP4_SHA)
    receipt=json.loads((original.parent/'EB01_ECHO_ACTUAL_RECEIPT.json').read_text())
    spec=pilot_spec('collaborator')
    if (receipt.get('studio_revision')!=SOURCE_REVISION or receipt.get('mp4_sha256')!=ORIGINAL_MP4_SHA or
        receipt.get('original_voice_sha256')!=spec['voice_sha256'] or receipt.get('success') is not True):
        raise ValueError('Frozen source pilot receipt changed')
    voice=find_original(mount,'S01.wav',spec['voice_sha256'])
    shutil.copyfile(original,out/'EB01_RESTORE_SOURCE_COMPARE_ONLY.mp4')
    finish=start(str(out/'EB01_RESTORE_RESOURCES_SAFE.json'));began=time.monotonic()
    try:
        print('EB01_STAGE_BEGIN_MODEL_LOAD',flush=True)
        weights=Path(hf_hub_download('alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP','Wan2.1_VAE.pth',
            revision=VAE_REVISION,local_dir=str(work/'vae'),token=False))
        if weights.stat().st_size!=507609880 or sha(weights)!=VAE_SHA:raise ValueError('Original VAE checkpoint mismatch')
        sys.path.insert(0,str(runtime))
        from src.wan_vae import AutoencoderKLWan
        vae=AutoencoderKLWan.from_pretrained(str(weights),additional_kwargs={
            'temporal_compression_ratio':4,'spatial_compression_ratio':8}).float().eval().to('cuda:0')
        latents=load_file(str(saved),device='cpu')['latents'].to(device='cuda:0',dtype=torch.float32)
        print('EB01_STAGE_BEGIN_DECODE',flush=True);decode_start=time.monotonic()
        with torch.inference_mode():
            sample=(vae.decode(latents).sample/2+.5).clamp(0,1)
        if tuple(sample.shape)!=(1,3,65,1024,576) or not torch.isfinite(sample).all():
            raise ValueError('Recovered native pixels invalid')
        frames=(sample[0].permute(1,2,3,0).cpu().float().numpy()*255).astype(np.uint8)
        decoded_seconds=round(time.monotonic()-decode_start,3)
        silent=out/'EB01_RESTORED_NATIVE_SILENT.mp4'
        with imageio.get_writer(silent,fps=25,codec='libx264',quality=9,macro_block_size=1) as writer:
            for frame in frames:writer.append_data(frame)
        final=out/'EB01_RESTORED_RUSSIAN_NATIVE.mp4'
        subprocess.run(['ffmpeg','-v','error','-i',str(silent),'-i',str(voice),'-map','0:v:0','-map','1:a:0',
            '-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-af','apad','-t','2.6',
            '-movflags','+faststart',str(final)],check=True,capture_output=True,timeout=120)
        resources=finish();finish=lambda:None
        proof.update(schema=1,status='FULL_ORIGINAL_LATENT_CHECKPOINT_DECODED_REVIEW_REQUIRED',
            success=True,source_pilot_run=SOURCE_RUN,source_pilot_revision=SOURCE_REVISION,
            source_original_mp4_sha256=ORIGINAL_MP4_SHA,restored_mp4_sha256=sha(final),
            original_voice_sha256=sha(voice),studio_revision=cfg['studio_revision'],
            upstream_revision=UPSTREAM_SHA,vae_weights_sha256=VAE_SHA,vae_revision=VAE_REVISION,
            native_tensor_shape=list(sample.shape),native_fps=25,native_width=576,native_height=1024,
            vae_precision='float32',vae_decode_seconds=decoded_seconds,
            process_wall_seconds=round(time.monotonic()-began,3),resources=resources,
            diffusion_model_loaded=False,text_encoder_loaded=False,acoustic_encoder_loaded=False,
            production_jobs_submitted=0,new_original_takes=0,synthetic_frames=0,source_video_upscaled=False,
            human_quality_approved=False,whole_movie_approved=False,published=False)
        (out/'EB01_RESTORE_RECEIPT_SAFE.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
        print('EB01_STAGE_NATIVE_VIDEO_OUTPUT_READY',flush=True)
    finally:finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);restore(p.parse_args().config)
