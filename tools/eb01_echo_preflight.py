"""Actual CPU import + public model accessibility proof before any GPU allocation."""
from __future__ import annotations
import argparse,base64,hashlib,importlib,json,os,subprocess,sys,tempfile
from pathlib import Path
from studio.eb01_echo_profile import UPSTREAM,UPSTREAM_SHA,REPOSITORIES,patch_upstream


def model_lock():
    from huggingface_hub import HfApi,get_hf_file_metadata,hf_hub_url,hf_hub_download
    api=HfApi(token=False);lock={}
    for alias,repo in REPOSITORIES.items():
        info=api.model_info(repo,files_metadata=True)
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',type=Path);args=parser.parse_args()
    os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
    root=args.runtime or Path(os.environ['RUNNER_TEMP'])/'eb01-echo-cpu-source'
    if not root.exists():subprocess.run(['git','clone','--depth','1',UPSTREAM,str(root)],check=True,capture_output=True,timeout=180)
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if actual!=UPSTREAM_SHA:
        subprocess.run(['git','-C',str(root),'fetch','--depth','1','origin',UPSTREAM_SHA],check=True,capture_output=True,timeout=120)
        subprocess.run(['git','-C',str(root),'checkout','--detach',UPSTREAM_SHA],check=True,capture_output=True)
    patch=patch_upstream(root);sys.path.insert(0,str(root))
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
    lock=model_lock()
    result={'schema':1,'upstream_sha':UPSTREAM_SHA,'cpu_imports_verified':True,
            'sdpa_padding_math_verified':True,'models':lock,'source_patch_sha256':patch,
            'gpu_jobs_submitted':0,'movie_approved':False}
    path=Path(os.environ['RUNNER_TEMP'])/'EB01_ECHO_CPU_PREFLIGHT_SAFE.json'
    path.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    packed=base64.b64encode(json.dumps(lock,sort_keys=True).encode()).decode()
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('model_lock_b64='+packed+'\n')
    print('EB01_ECHO_CPU_IMPORT_AND_MODEL_METADATA_PASS',json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
