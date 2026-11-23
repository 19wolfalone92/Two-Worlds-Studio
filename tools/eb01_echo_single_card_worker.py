"""Real private single-T4 inference, exact original-voice chunks and tail PNGs."""
from __future__ import annotations
import argparse,gc,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def load_vision(base,flash,runtime):
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel
    from src.wan_vae import AutoencoderKLWan
    from src.wan_image_encoder import CLIPModel
    from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from src.utils import filter_kwargs
    from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
    conf=OmegaConf.load(runtime/'config/config.yaml')
    model_cfg=json.loads((base/'config.json').read_text())
    extra=OmegaConf.to_container(conf.transformer_additional_kwargs)
    for original,mapped in extra.pop('dict_mapping').items():extra[mapped]=model_cfg[original]
    extra.pop('transformer_subpath',None)
    with init_empty_weights():transformer=WanTransformerAudioMask3DModel.from_config(model_cfg,**extra)
    with safe_open(flash/'diffusion_pytorch_model.safetensors',framework='pt',device='cpu') as checkpoint:
        expected=transformer.state_dict()
        if set(expected)!=set(checkpoint.keys()):raise ValueError('Incomplete Flash architecture')
        for name in checkpoint.keys():
            value=checkpoint.get_tensor(name)
            if tuple(value.shape)!=tuple(expected[name].shape):raise ValueError('Wrong Flash tensor shape')
            set_module_tensor_to_device(transformer,name,'cpu',value=value,dtype=torch.float16)
    transformer.eval();gc.collect()
    vae=AutoencoderKLWan.from_pretrained(str(base/conf.vae_kwargs.vae_subpath),
        additional_kwargs=OmegaConf.to_container(conf.vae_kwargs)).float().eval()
    clip=CLIPModel.from_pretrained(str(base/conf.image_encoder_kwargs.image_encoder_subpath)).half().eval()
    args=OmegaConf.to_container(conf.scheduler_kwargs);args['shift']=1
    scheduler=FlowUniPCMultistepScheduler(**filter_kwargs(FlowUniPCMultistepScheduler,args))
    return WanFunInpaintAudioPipeline(tokenizer=None,text_encoder=None,vae=vae,
        transformer=transformer,clip_image_encoder=clip,scheduler=scheduler)


def run(config,slot):
    import torch,numpy as np,soundfile as sf,librosa
    from PIL import Image
    from safetensors.torch import load_file
    import imageio.v2 as imageio
    from studio.eb01_echo_parallel_profile import upgrade_precomputed
    from studio.eb01_anatomy_profile import select_specs,patch_anatomy,native_audio_mask_numpy,constrain_reference
    from studio.eb01_echo_profile import patch_upstream
    from studio.eb01_v7_storyboard import gather_private
    from tools.eb01_echo_native_infer import sha,make_native_reference
    from tools.eb01_resource_sampler import start
    cfg=json.loads(Path(config).read_text());spec=select_specs(cfg)[slot]
    if torch.cuda.device_count()!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(slot):
        raise RuntimeError('Worker must see exactly its one assigned physical GPU')
    root=Path(cfg['work']);work=root/('worker'+str(slot));work.mkdir(exist_ok=True)
    out=Path('/kaggle/working')/('slot'+str(slot));out.mkdir(exist_ok=True)
    runtime=work/'runtime'
    import shutil
    shutil.copytree(cfg['runtime'],runtime)
    patch_upstream(runtime);patch=upgrade_precomputed(runtime)
    if 'anatomy_controls' in spec:patch=patch_anatomy(runtime)
    sys.path.insert(0,str(runtime))
    finish=start(str(out/'EB01_ECHO_RESOURCES_SAFE.json'));started=time.monotonic()
    try:
        print('EB01_STAGE_BEGIN_MODEL_LOAD',flush=True)
        # The public weights were already pinned, downloaded and SHA-checked
        # once. Recheck vision/audio bytes independently; do not reload T5.
        hashes=json.loads((root/'PUBLIC_MODEL_HASHES_SAFE.json').read_text())
        for key,expected in hashes.items():
            if key.startswith('base/models_t5_') or key.startswith('base/google/'):continue
            if sha(root/'models'/key)!=expected:raise ValueError('Shared public weight checksum mismatch')
        private=work/'canonical';gather_private(Path('/kaggle/input'),private)
        voice=private/'voices'/(spec['shot']+'.wav')
        waveform,sr=sf.read(voice,dtype='float32',always_2d=True);waveform=waveform.mean(axis=1)
        if sha(voice)!=spec['voice_sha256'] or abs(len(waveform)/sr-spec['voice_seconds'])>.015:
            raise ValueError('Original full Russian voice checksum or clock mismatch')
        duration=spec['frames']/25
        if duration+1e-6<len(waveform)/sr+.2:raise ValueError('Original voice would be truncated')
        audio=librosa.resample(waveform,orig_sr=sr,target_sr=16000)
        audio=np.pad(audio,(0,round(duration*16000)-len(audio)))
        from src.wav2vec2 import Wav2Vec2Model
        from transformers import Wav2Vec2FeatureExtractor
        enc=Wav2Vec2Model.from_pretrained(str(root/'models/audio'),local_files_only=True).cpu().eval()
        ext=Wav2Vec2FeatureExtractor.from_pretrained(str(root/'models/audio'),local_files_only=True)
        features=ext(audio,sampling_rate=16000,return_tensors='pt').input_values
        with torch.inference_mode():encoded=enc(features,seq_len=spec['frames'],output_hidden_states=True)
        acoustic=torch.stack(encoded.hidden_states[1:],dim=1).squeeze(0).permute(1,0,2)
        indices=(torch.arange(spec['frames'])[:,None]+(torch.arange(5)-2)[None,:]).clamp(0,spec['frames']-1)
        acoustic=acoustic[indices][None].half().contiguous()
        del enc,ext,features,encoded;gc.collect()
        key=spec.get('context_key',spec['shot']);text_path=root/'text-contexts'/(key+'.safetensors')
        context=json.loads((root/'text-contexts'/(key+'.json')).read_text())
        if (sha(text_path)!=context['embeddings_sha256'] or context['prompt_sha256']!=spec['prompt_sha256'] or
            context['source_revision']!=cfg['studio_revision']):raise ValueError('Precomputed text context mismatch')
        prompt=load_file(str(text_path))['prompt_embeds'].to('cuda:0')
        if not torch.isfinite(prompt).all():raise ValueError('Invalid precomputed text embeddings')
        reference=work/'native-reference.png';make_native_reference(Path('/kaggle/input'),reference,spec)
        first_reference_sha=sha(reference)
        canonical_frame = None
        if spec.get('anatomy_controls', {}).get('static_reference_boxes_xyxy'):
            with Image.open(reference) as original:
                array = np.array(original.convert('RGB'))
            canonical_frame = torch.from_numpy(array).permute(2, 0, 1)[None, :, None].float() / 255
        pipeline=load_vision(root/'models/base',root/'models/flash/echomimicv3-flash-pro',runtime)
        from src.utils import get_image_to_video_latent2
        (out/'WORKER_MODEL_READY_SAFE.json').write_text(json.dumps({'slot':slot,'model_ready':True}))
        print('EB01_STAGE_END_MODEL_LOAD',flush=True)
        generated_at=time.monotonic();offset=0;segments=[];native_seconds=0.
        for index,frames in enumerate(spec['segment_frames']):
            input_sha=sha(reference)
            image=Image.open(reference).convert('RGB')
            video,mask,clip_image=get_image_to_video_latent2(image,None,frames,[1024,576])
            spatial_audio=None
            if 'anatomy_controls' in spec:
                if canonical_frame is not None:
                    constrain_reference(video, mask, canonical_frame, spec)
                spatial_audio = torch.from_numpy(native_audio_mask_numpy(spec, frames)).to('cuda:0')
            checkpoint=out/('SEGMENT_'+str(index)+'_LATENTS.safetensors')
            os.environ['EB01_ECHO_LATENT_CHECKPOINT']=str(checkpoint)
            embeddings=acoustic[:,offset:offset+frames].to('cuda:0')
            if embeddings.shape[1]!=frames:raise ValueError('Original audio context slice truncated')
            segment_started=time.monotonic()
            with torch.inference_mode():
                sample=pipeline(prompt=None,prompt_embeds=prompt,negative_prompt_embeds=prompt,
                    negative_prompt=None,num_frames=frames,audio_embeds=embeddings,audio_scale=1.,
                    ip_mask=spatial_audio,use_un_ip_mask=False,height=1024,width=576,
                    generator=torch.Generator(device='cuda:0').manual_seed(spec['seed']+index),
                    neg_scale=1.,neg_steps=0,use_dynamic_cfg=False,use_dynamic_acfg=False,
                    guidance_scale=6.,audio_guidance_scale=spec.get('anatomy_controls',{}).get('audio_guidance_scale',3.),num_inference_steps=8,
                    video=video,mask_video=mask,clip_image=clip_image,cfg_skip_ratio=0.,shift=5.).videos
            segment_seconds=time.monotonic()-segment_started;native_seconds+=segment_seconds
            shape=list(sample.shape)
            if shape!=[1,3,frames,1024,576] or not torch.isfinite(sample).all():raise ValueError('Wrong original model pixels')
            target=out/('SEGMENT_'+str(index)+'_NATIVE.mp4')
            pixels=(sample[0].permute(1,2,3,0).cpu().float().clamp(0,1).numpy()*255).astype('uint8')
            with imageio.get_writer(target,fps=25,codec='libx264',quality=9,macro_block_size=1) as writer:
                for frame in pixels:writer.append_data(frame)
            tail=out/('SEGMENT_'+str(index)+'_DECODED_TAIL.png')
            subprocess.run(['ffmpeg','-v','error','-i',str(target),'-vf',f'select=eq(n\\,{frames-1})',
                '-vsync','0','-frames:v','1',str(tail)],check=True,capture_output=True,timeout=90)
            with Image.open(tail) as im:tail_rgb_sha=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            segments.append({'index':index,'native_frames':frames,'actual_native_tensor_shape':shape,
                'native_inference_seconds':round(segment_seconds,3),
                'start_scene_frame':offset,'audio_conditioning_embeddings_shape':list(embeddings.shape),
                'conditioning_png_sha256':input_sha,'mp4_sha256':sha(target),
                'last_decoded_rgb_sha256':tail_rgb_sha,'next_conditioning_png_sha256':sha(tail),
                'spatial_audio_mask_sha256':hashlib.sha256(spatial_audio.detach().cpu().numpy().tobytes()).hexdigest() if spatial_audio is not None else None,
                'spatial_audio_mask_shape':list(spatial_audio.shape) if spatial_audio is not None else None})
            reference=tail;offset+=frames-1
            del sample,pixels,embeddings,video,mask,spatial_audio;gc.collect();torch.cuda.empty_cache()
        joined=out/'EB01_ECHO_NATIVE_SILENT.mp4';join_proof=None
        if len(segments)==1:shutil.copy2(out/'SEGMENT_0_NATIVE.mp4',joined)
        else:
            from tools.eb01_echo_lossless_assembly import join_video
            join_proof=join_video([out/('SEGMENT_'+str(i)+'_NATIVE.mp4') for i in range(len(segments))],joined)
        final=out/'EB01_ECHO_RUSSIAN_NATIVE.mp4'
        subprocess.run(['ffmpeg','-v','error','-i',str(joined),'-i',str(voice),'-map','0:v:0','-map','1:a:0',
            '-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-af','apad','-t',str(duration),
            '-video_track_timescale','25','-movflags','+faststart',str(final)],check=True,capture_output=True,timeout=90)
        resources=finish();finish=lambda:None
        receipt={**spec,'schema':1,'studio_revision':cfg['studio_revision'],'upstream_revision':cfg['upstream_revision'],
            'source_patch_sha256':patch,'first_reference_sha256':first_reference_sha,'text_context_sha256':sha(text_path),
            'precomputed_text_encoder_released_before_inference':True,'model_weights_sha256':hashes,
            'original_voice_sha256':sha(voice),'original_voice_truncated':False,'segments':segments,
            'lossless_native_join_verification':join_proof,
            'audio_conditioning_embeddings_shape':list(acoustic.shape),'mp4_sha256':sha(final),
            'native_inference_seconds':round(native_seconds,3),
            'generation_encoding_join_wall_seconds':round(time.monotonic()-generated_at,3),
            'process_wall_seconds':round(time.monotonic()-started,3),'resources':resources,
            'source_video_upscaled':False,'synthetic_frames':0,'held_frames':0,
            'overlap_frames_discarded':len(segments)-1,'native_video_generated':True,
            'human_artistic_approved':False,'lipsync_quality_approved':False,'whole_movie_approved':False,'success':True}
        (out/'EB01_ECHO_ACTUAL_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
        print('EB01_STAGE_NATIVE_VIDEO_OUTPUT_READY',flush=True)
    finally:finish()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--slot',type=int,choices=(0,1),required=True)
    a=p.parse_args();run(a.config,a.slot)
