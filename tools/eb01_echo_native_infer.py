"""Actual original EchoMimic V3 Flash forward with FULL canonical Russian audio.

Private Kaggle process only. No face patch, new portrait, video upscale,
frame interpolation, paid inference, social publishing or artistic self-approval.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,math,os,shutil,subprocess,sys,time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


def download_locked(lock,destination):
    from huggingface_hub import hf_hub_download
    destination.mkdir(parents=True,exist_ok=True)
    result={}
    for alias,entry in lock.items():
        if not __import__('re').fullmatch('[0-9a-f]{40}',entry['revision']):raise ValueError('Unpinned model revision')
        root=destination/alias;root.mkdir(exist_ok=True)
        for item in entry['files']:
            if Path(item['path']).is_absolute() or '..' in Path(item['path']).parts:raise ValueError('Unsafe model path')
            path=Path(hf_hub_download(entry['repo_id'],item['path'],revision=entry['revision'],
                local_dir=str(root),token=False))
            if path.stat().st_size!=item['size']:raise ValueError('Official checkpoint byte size mismatch')
            expected=item.get('lfs_sha256')
            actual=sha(path)
            if expected:
                if actual!=expected:raise ValueError('Official checkpoint LFS SHA mismatch')
            else:
                raw=path.read_bytes()
                blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
                if blob!=item['blob_id']:raise ValueError('Official model config Git blob mismatch')
            result[alias+'/'+item['path']]=actual
    return result


def make_native_reference(mount,output,spec):
    from PIL import Image,ImageOps,ImageEnhance
    from studio.eb01_v7_storyboard import SHOTS
    who=spec['actor'];paths=[p for p in mount.rglob(who+'.png') if sha(p)==spec['portrait_sha256']]
    if len(paths)!=1:raise ValueError('Canonical full-resolution portrait missing or ambiguous')
    scene=next(s for s in SHOTS if s[0]==spec['shot']);zoom,cx,cy=scene[4:7]
    with Image.open(paths[0]) as im:
        if im.size!=(704,1216) or im.format!='PNG':raise ValueError('Original portrait geometry changed')
        iw,ih=im.size;rw,rh=int(iw/zoom),int(ih/zoom)
        if rw<spec['width'] or rh<spec['height']:raise ValueError('Original has insufficient native pixels')
        x=max(0,min(iw-rw,int((iw-rw)*cx*2)));y=max(0,min(ih-rh,int((ih-rh)*cy*2)))
        im=ImageOps.fit(im.convert('RGB').crop((x,y,x+rw,y+rh)),(spec['width'],spec['height']),Image.Resampling.LANCZOS)
        im=ImageEnhance.Color(im).enhance(.98);im=ImageEnhance.Contrast(im).enhance(1.03)
        im.save(output)
    return sha(output)


def infer(config_path):
    import numpy as np
    import torch
    import soundfile as sf
    import librosa
    from PIL import Image
    from omegaconf import OmegaConf
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoTokenizer,Wav2Vec2FeatureExtractor
    from safetensors import safe_open
    from studio.eb01_echo_profile import pilot_spec,patch_upstream,PROFILE,REPOSITORIES
    from studio.eb01_v7_storyboard import gather_private,SHOTS
    from tools.eb01_resource_sampler import start
    cfg=json.loads(Path(config_path).read_text());spec=pilot_spec(cfg['account_alias'])
    runtime=Path(cfg['runtime']);work=Path(cfg['work']);out=Path('/kaggle/working')
    if (not torch.cuda.is_available() or torch.cuda.device_count()!=2 or
        any(torch.cuda.get_device_properties(i).total_memory<14*2**30 for i in (0,1))):
        raise RuntimeError('Two separate real >=14 GiB CUDA devices required')
    lock=cfg['model_lock']
    if set(lock)!=set(REPOSITORIES) or any(lock[k]['repo_id']!=REPOSITORIES[k] for k in lock):
        raise ValueError('Unapproved checkpoint source')
    finish=start(str(out/'EB01_ECHO_RESOURCES_SAFE.json'))
    begin=time.monotonic()
    try:
        patch=patch_upstream(runtime);sys.path.insert(0,str(runtime))
        print('EB01_STAGE_BEGIN_MODEL_LOAD',flush=True)
        # Stage checkpoint disk use as well as VRAM: T5 is ~10.6 GiB by
        # itself. Verify/load it on card 1 and release its disk copy before
        # fetching the vision weights, so a 20 GiB scratch volume can work.
        text_file='models_t5_umt5-xxl-enc-bf16.pth'
        early={'audio':lock['audio'],'base':{**lock['base'],'files':[
            f for f in lock['base']['files'] if f['path']==text_file or f['path'].startswith('google/')]}}
        later={'flash':lock['flash'],'base':{**lock['base'],'files':[
            f for f in lock['base']['files'] if f['path']!=text_file and not f['path'].startswith('google/')]}}
        model_hashes=download_locked(early,work/'models')
        bases=work/'models/base';flash=work/'models/flash/echomimicv3-flash-pro';audio=work/'models/audio'
        private=work/'canonical';gather_private(Path('/kaggle/input'),private)
        original_voice=private/'voices'/(spec['shot']+'.wav')
        if sha(original_voice)!=spec['voice_sha256']:raise ValueError('Original Russian WAV SHA mismatch')
        waveform,sr=sf.read(original_voice,dtype='float32',always_2d=True)
        waveform=waveform.mean(axis=1)
        original_seconds=len(waveform)/sr
        if abs(original_seconds-spec['voice_seconds'])>.015:raise ValueError('Original complete voice clock mismatch')
        duration=spec['frames']/spec['fps']
        if duration<original_seconds+.2:raise ValueError('Refuse truncating original spoken line')
        resampled=librosa.resample(waveform,orig_sr=sr,target_sr=16000)
        target=round(duration*16000)
        if len(resampled)>target:raise ValueError('Original speech would be cut')
        acoustic=np.pad(resampled,(0,target-len(resampled)))
        reference=work/'native-reference.png';ref_sha=make_native_reference(Path('/kaggle/input'),reference,spec)
        from src.wav2vec2 import Wav2Vec2Model
        audio_encoder=Wav2Vec2Model.from_pretrained(str(audio),local_files_only=True).cpu().eval()
        extractor=Wav2Vec2FeatureExtractor.from_pretrained(str(audio),local_files_only=True)
        features=extractor(acoustic,sampling_rate=16000,return_tensors='pt').input_values
        with torch.inference_mode():
            encoded=audio_encoder(features,seq_len=spec['frames'],output_hidden_states=True)
        embeddings=torch.stack(encoded.hidden_states[1:],dim=1).squeeze(0).permute(1,0,2)
        offsets=torch.arange(5)-2
        indices=(torch.arange(spec['frames'])[:,None]+offsets[None,:]).clamp(0,spec['frames']-1)
        embeddings=embeddings[indices][None].to(device='cuda:0',dtype=torch.float16)
        del audio_encoder,encoded,features;gc.collect()
        configuration=OmegaConf.load(runtime/'config/config.yaml')
        from src.wan_text_encoder import WanT5EncoderModel
        # Dedicated second card, not pooled VRAM. Load T5 first so the CPU
        # copy is released before the larger vision modules are allocated.
        text_encoder=WanT5EncoderModel.from_pretrained(str(bases/configuration.text_encoder_kwargs.text_encoder_subpath),
            additional_kwargs=OmegaConf.to_container(configuration.text_encoder_kwargs),
            low_cpu_mem_usage=True,torch_dtype=torch.float16).half().eval().to('cuda:1')
        if any(p.is_meta for p in text_encoder.parameters()):raise ValueError('Incomplete T5 checkpoint load')
        (bases/text_file).unlink()
        model_hashes.update(download_locked(later,work/'models'))
        (out/'EB01_ECHO_MODEL_SHA_SAFE.json').write_text(json.dumps(model_hashes,sort_keys=True))
        tokenizer=AutoTokenizer.from_pretrained(str(bases/configuration.text_encoder_kwargs.tokenizer_subpath),local_files_only=True)
        from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel
        # Load the actual Flash checkpoint directly into the exact architecture;
        # do not initialize a base model and silently ignore missing audio keys.
        model_cfg=json.loads((bases/'config.json').read_text())
        extra=OmegaConf.to_container(configuration.transformer_additional_kwargs)
        for original,mapped in extra.pop('dict_mapping').items():extra[mapped]=model_cfg[original]
        extra.pop('transformer_subpath',None)
        with init_empty_weights():transformer=WanTransformerAudioMask3DModel.from_config(model_cfg,**extra)
        with safe_open(flash/'diffusion_pytorch_model.safetensors',framework='pt',device='cpu') as checkpoint:
            expected=transformer.state_dict()
            if set(expected)!=set(checkpoint.keys()):raise ValueError('Flash weights do not completely match audio architecture')
            for name in checkpoint.keys():
                value=checkpoint.get_tensor(name)
                if tuple(value.shape)!=tuple(expected[name].shape):raise ValueError('Flash checkpoint tensor shape mismatch')
                set_module_tensor_to_device(transformer,name,'cpu',value=value,dtype=torch.float16)
        transformer.eval();gc.collect()
        from src.wan_vae import AutoencoderKLWan
        vae=AutoencoderKLWan.from_pretrained(str(bases/configuration.vae_kwargs.vae_subpath),
            additional_kwargs=OmegaConf.to_container(configuration.vae_kwargs)).float().eval()
        from src.wan_image_encoder import CLIPModel
        clip=CLIPModel.from_pretrained(str(bases/configuration.image_encoder_kwargs.image_encoder_subpath)).half().eval()
        from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from src.utils import filter_kwargs,get_image_to_video_latent2
        from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
        scheduler_cfg=OmegaConf.to_container(configuration.scheduler_kwargs);scheduler_cfg['shift']=1
        scheduler=FlowUniPCMultistepScheduler(**filter_kwargs(FlowUniPCMultistepScheduler,scheduler_cfg))
        pipeline=WanFunInpaintAudioPipeline(tokenizer=tokenizer,text_encoder=text_encoder,vae=vae,
            transformer=transformer,clip_image_encoder=clip,scheduler=scheduler)
        # No pipeline.to(cuda): staging is explicit in the guarded upstream patch.
        # No cache/quantization changes in this first controlled audio pilot.
        image=Image.open(reference).convert('RGB')
        input_video,mask_video,clip_image=get_image_to_video_latent2(image,None,spec['frames'],[spec['height'],spec['width']])
        os.environ['EB01_ECHO_LATENT_CHECKPOINT']=str(out/'EB01_ECHO_DENOISED_LATENTS.safetensors')
        prompt=next(s[-1] for s in SHOTS if s[0]==spec['shot'])
        print('EB01_STAGE_END_MODEL_LOAD',flush=True)
        generated_at=time.monotonic()
        with torch.inference_mode():
            sample=pipeline(prompt,num_frames=spec['frames'],negative_prompt='',audio_embeds=embeddings,
                audio_scale=1.0,ip_mask=None,use_un_ip_mask=False,height=spec['height'],width=spec['width'],
                generator=torch.Generator(device='cuda:0').manual_seed(spec['seed']),neg_scale=1.,neg_steps=0,
                use_dynamic_cfg=False,use_dynamic_acfg=False,guidance_scale=6.,audio_guidance_scale=3.,
                num_inference_steps=spec['steps'],video=input_video,mask_video=mask_video,
                clip_image=clip_image,cfg_skip_ratio=0.,shift=5.).videos
        inference_seconds=round(time.monotonic()-generated_at,3)
        if tuple(sample.shape)!=(1,3,spec['frames'],spec['height'],spec['width']):raise ValueError('Model generated wrong native dimensions or frame count')
        if not torch.isfinite(sample).all():raise ValueError('Nonfinite native pixels')
        # Encode exactly each original model frame, with no global color-transfer
        # postprocess, face restoration, resize, fps filter or interpolation.
        frames=(sample[0].permute(1,2,3,0).cpu().float().clamp(0,1).numpy()*255).astype('uint8')
        video=out/'EB01_ECHO_NATIVE_SILENT.mp4'
        import imageio.v2 as imageio
        with imageio.get_writer(video,fps=25,codec='libx264',quality=9,macro_block_size=1) as writer:
            for frame in frames:writer.append_data(frame)
        final=out/'EB01_ECHO_RUSSIAN_NATIVE.mp4'
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-i',str(original_voice),
            '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000',
            '-af','apad','-t',str(duration),'-movflags','+faststart',str(final)],check=True,capture_output=True,timeout=120)
        resource=finish();finish=lambda:None
        proof={'schema':1,**spec,'studio_revision':cfg['studio_revision'],'upstream_revision':cfg['upstream_revision'],
            'source_patch_sha256':patch,'model_revisions':{k:v['revision'] for k,v in lock.items()},
            'model_weights_sha256':model_hashes,'input_reference_sha256':ref_sha,
            'original_voice_sha256':sha(original_voice),'original_voice_seconds':original_seconds,
            'acoustic_input_samples':len(acoustic),'acoustic_input_sample_rate':16000,
            'original_voice_truncated':False,'audio_conditioning_embeddings_shape':list(embeddings.shape),
            'audio_conditioning_embeddings_sha256':hashlib.sha256(embeddings.detach().cpu().numpy().tobytes()).hexdigest(),
            'mp4_sha256':sha(final),'native_silent_mp4_sha256':sha(video),
            'actual_native_tensor_shape':list(sample.shape),'native_inference_seconds':inference_seconds,
            'process_wall_seconds':round(time.monotonic()-begin,3),'resources':resource,
            'source_video_upscaled':False,'synthetic_frames':0,'face_only_patch':False,
            'native_video_generated':True,'audio_is_model_input':True,
            'human_artistic_approved':False,'lipsync_quality_approved':False,'whole_movie_approved':False,
            'published':False,'success':True,'status':'REAL_AUDIO_DRIVEN_NATIVE_MP4_CREATED_REVIEW_REQUIRED'}
        (out/'EB01_ECHO_ACTUAL_RECEIPT.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
        print('EB01_STAGE_NATIVE_VIDEO_OUTPUT_READY',flush=True)
    finally:finish()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    infer(p.parse_args().config)
