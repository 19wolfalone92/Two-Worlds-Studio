"""Actual CPU import + public model accessibility proof before any GPU allocation."""
from __future__ import annotations
import argparse,base64,hashlib,importlib,json,os,subprocess,sys,tempfile
from pathlib import Path
from studio.eb01_echo_profile import UPSTREAM,UPSTREAM_SHA,REPOSITORIES,patch_upstream


def model_lock(revisions=None):
    from huggingface_hub import HfApi,get_hf_file_metadata,hf_hub_url,hf_hub_download
    api=HfApi(token=False);lock={}
    for alias,repo in REPOSITORIES.items():
        info=api.model_info(repo,revision=revisions[alias] if revisions else None,files_metadata=True)
        if not info.sha or len(info.sha)!=40:raise ValueError('Model commit unresolved')
        files=[]
        for item in info.siblings:
            name=item.rfilename
            selected=(alias=='base' and (name in ('config.json','Wan2.1_VAE.pth','models_t5_umt5-xxl-enc-bf16.pth','models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth') or name.startswith('google/umt5-xxl/')) or
                      alias=='flash' and name in ('echomimicv3-flash-pro/config.json','echomimicv3-flash-pro/diffusion_pytorch_model.safetensors') or
                      alias=='audio' and name in ('config.json','preprocessor_config.json','pytorch_model.bin'))
            if not selected:continue
            lfs=getattr(item,'lfs',None)
            sha=getattr(lfs,'sha256',None) if lfs else None
            files.append({'path':name,'size':item.size,'lfs_sha256':sha,'blob_id':item.blob_id})
        if len(files)<(4 if alias=='base' else 2):raise ValueError('Incomplete official checkpoint manifest')
        for f in files:
            # HEAD checks actual pinned file and its redirected storage without
            # downloading tens of GB on a CPU runner.
            get_hf_file_metadata(hf_hub_url(repo,f['path'],revision=info.sha),token=False,timeout=60)
        lock[alias]={'repo_id':repo,'revision':info.sha,'files':files}
    return lock


def real_acoustic_encoder_check(lock, root):
    """Real public pretrained encoder + real resampling; no private user speech."""
    import numpy as np
    import torch
    import librosa
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2FeatureExtractor
    from src.wav2vec2 import Wav2Vec2Model
    entry=lock['audio'];model_dir=root/'public-audio-encoder'
    for item in entry['files']:
        path=Path(hf_hub_download(entry['repo_id'],item['path'],revision=entry['revision'],
                                local_dir=str(model_dir),token=False))
        if item.get('lfs_sha256'):
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=item['lfs_sha256']:raise ValueError('Real CPU audio encoder weights SHA mismatch')
    source=(.03*np.sin(2*np.pi*220*np.arange(24000)/24000)).astype('float32')
    waveform=librosa.resample(source,orig_sr=24000,target_sr=16000)
    extractor=Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir),local_files_only=True)
    encoder=Wav2Vec2Model.from_pretrained(str(model_dir),local_files_only=True).cpu().eval()
    features=extractor(waveform,sampling_rate=16000,return_tensors='pt').input_values
    with torch.inference_mode():result=encoder(features,seq_len=25,output_hidden_states=True)
    states=result.hidden_states[1:]
    if len(states)!=12 or any(tuple(x.shape)!=(1,25,768) or not torch.isfinite(x).all() for x in states):
        raise ValueError('Original pretrained acoustic conditioning has wrong clock or nonfinite values')
    return {'real_pretrained_audio_encoder_cpu_forward':True,'resampling_24000_to_16000_verified':True,
            'hidden_state_layers':12,'native_frames':25,'hidden_width':768,
            'input_is_public_synthetic_fixture_not_private_user_voice':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',type=Path);parser.add_argument('--precomputed-contract',action='store_true')
    parser.add_argument('--anatomy-contract',action='store_true');args=parser.parse_args()
    os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
    root=args.runtime or Path(os.environ['RUNNER_TEMP'])/'eb01-echo-cpu-source'
    if not root.exists():subprocess.run(['git','clone','--depth','1',UPSTREAM,str(root)],check=True,capture_output=True,timeout=180)
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if actual!=UPSTREAM_SHA:
        subprocess.run(['git','-C',str(root),'fetch','--depth','1','origin',UPSTREAM_SHA],check=True,capture_output=True,timeout=120)
        subprocess.run(['git','-C',str(root),'checkout','--detach',UPSTREAM_SHA],check=True,capture_output=True)
    patch=patch_upstream(root)
    if args.precomputed_contract:
        from studio.eb01_echo_parallel_profile import upgrade_precomputed
        patch['src/pipeline_wan_fun_inpaint_audio_2512.py']=upgrade_precomputed(root)
    if args.anatomy_contract:
        if not args.precomputed_contract:raise ValueError('Anatomy check requires actual precomputed-context contract')
        from studio.eb01_anatomy_profile import patch_anatomy
        patch.update(patch_anatomy(root))
    sys.path.insert(0,str(root))
    for name in ('src.wav2vec2','src.wan_vae','src.wan_image_encoder','src.wan_text_encoder',
                 'src.wan_transformer3d_audio_2512','src.pipeline_wan_fun_inpaint_audio_2512'):
        importlib.import_module(name)
    import torch
    from src.wan_transformer3d_audio_2512 import attention
    # This checks padding against explicit math, not merely function presence.
    q=torch.randn(1,3,2,8);k=torch.randn(1,5,2,8);v=torch.randn(1,5,2,8)
    got=attention(q,k,v,k_lens=torch.tensor([3]))
    wanted=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k[:,:3].transpose(1,2),v[:,:3].transpose(1,2)).transpose(1,2)
    if not torch.allclose(got,wanted,atol=1e-5,rtol=1e-5):raise ValueError('SDPA padding changes model semantics')
    if args.anatomy_contract:
        from src.wan_transformer3d_audio_2512 import WanI2VCrossAttentionAudio
        layer=WanI2VCrossAttentionAudio(16,2).float().eval()
        x=torch.randn(1,12,16);context=torch.randn(1,260,16);audio=torch.randn(1,2,4,16)
        zero=torch.zeros(1,12,1,1);mask=zero.clone();mask[:,4:8]=1
        with torch.inference_mode():
            no_audio=layer(x,(context,audio,2,zero,None),None,torch.float32)
            full=layer(x,(context,audio,2,None,None),None,torch.float32)
            part=layer(x,(context,audio,2,mask,None),None,torch.float32)
        expected=no_audio+(full-no_audio)*mask.view(1,12,1)
        if not torch.allclose(part,expected,atol=1e-5,rtol=1e-5) or torch.allclose(full,no_audio):
            raise ValueError('Actual upstream forward ignores or misapplies anatomical audio mask')
    if args.precomputed_contract:
        from accelerate import init_empty_weights
        from src.wan_vae import AutoencoderKLWan
        from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
        from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
        with init_empty_weights():vae=AutoencoderKLWan()
        pipe=WanFunInpaintAudioPipeline(tokenizer=None,text_encoder=None,vae=vae,
            transformer=None,clip_image_encoder=None,scheduler=FlowUniPCMultistepScheduler())
        context=torch.zeros(1,3,4096,dtype=torch.float16)
        actual,negative=pipe.encode_prompt(prompt=None,negative_prompt=None,
            prompt_embeds=context,negative_prompt_embeds=context,
            do_classifier_free_guidance=True,device=torch.device('cpu'))
        if actual is not context or negative is not context:raise ValueError('Precomputed pipeline changes supplied context')
        del pipe,vae
    from studio.eb01_echo_production_manifest import MODEL_REVISIONS
    lock=model_lock(MODEL_REVISIONS if args.anatomy_contract else None)
    if args.anatomy_contract:
        if {k:v['revision'] for k,v in lock.items()}!=MODEL_REVISIONS:raise ValueError('Anatomy comparison must retain all previously verified model revisions')
    acoustic=real_acoustic_encoder_check(lock,root.parent)
    result={'schema':1,'upstream_sha':UPSTREAM_SHA,'cpu_imports_verified':True,
            'sdpa_padding_math_verified':True,'real_acoustic_encoder':acoustic,'models':lock,'source_patch_sha256':patch,
            'precomputed_context_without_live_text_encoder_verified':args.precomputed_contract,
            'real_upstream_anatomical_audio_mask_forward_verified':args.anatomy_contract,
            'gpu_jobs_submitted':0,'movie_approved':False}
    path=Path(os.environ['RUNNER_TEMP'])/'EB01_ECHO_CPU_PREFLIGHT_SAFE.json'
    path.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    packed=base64.b64encode(json.dumps(lock,sort_keys=True).encode()).decode()
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('model_lock_b64='+packed+'\n')
    print('EB01_ECHO_CPU_IMPORT_AND_MODEL_METADATA_PASS',json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
