"""T-Flex 200 mm sculpt project: private Kaggle accelerator PROBE only.
Does not read original user screenshots; publishes hardware status only.
Never prints or stores auth tokens; does not run image generation.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

def identity():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from studio.kaggle_identity import resolve_username
    return resolve_username(None)

def prepare():
    who=identity()
    root=Path('dist/tflex_v9_gpu_probe');root.mkdir(parents=True,exist_ok=True)
    code='''import json, os, platform, time
from pathlib import Path
import torch
p=Path('/kaggle/working/tflex_v9_gpu_proof.json')
devices=[]
for i in range(torch.cuda.device_count()):
    x=torch.cuda.get_device_properties(i)
    devices.append({"ordinal":i,"name":x.name,"total_vram_gib":round(x.total_memory/2**30,2)})
report={"project":"T-Flex V9 200mm","gpu_count":len(devices),"devices":devices,
        "torch_version":torch.__version__,"cuda_version":torch.version.cuda,
        "private_user_images_received":False,"generated_3d_master":False}
p.write_text(json.dumps(report,indent=2))
print("TFLEX_PRIVATE_GPU_CAPABILITY:",json.dumps(report))
if len(devices)<1: raise RuntimeError("No Kaggle GPU assigned")
'''
    (root/'probe.py').write_text(code,encoding='utf8')
    meta={"id":f"{who}/tflex-v9-private-3d-gpu-probe","title":"TFlex V9 private 3D GPU probe",
          "code_file":"probe.py","language":"python","kernel_type":"script",
          "is_private":True,"enable_gpu":True,"enable_internet":False,
          "machine_shape":"NvidiaTeslaT4","dataset_sources":[],"competition_sources":[],
          "kernel_sources":[],"model_sources":[]}
    (root/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print("PRIVATE_KERNEL_CONFIG_READY",who,"images=NONE")
    return meta['id']

def poll(kernel,limit=1100):
    deadline=time.monotonic()+limit;last=None
    while time.monotonic()<deadline:
        r=subprocess.run(['kaggle','kernels','status',kernel],capture_output=True,text=True,timeout=90)
        if r.returncode==0:
            # Status CLI output contains no secrets; only emit normalized state
            message=r.stdout.lower()
            state=next((s for s in ['complete','error','failed','cancelled','running','queued','pending'] if s in message),'unknown')
            if state!=last:print('KAGGLE_KERNEL_STATE',state,flush=True);last=state
            if state=='complete':return True
            if state in ['error','failed','cancelled']:raise RuntimeError(f'Kaggle kernel ended {state}')
        else:
            print('KAGGLE_STATUS_API_UNAVAILABLE',r.returncode,flush=True)
            if ('429' in r.stderr or 'quota' in r.stderr.lower()):
                raise RuntimeError('Kaggle API rate/quota limited; not bypassing limits')
        time.sleep(70)
    raise TimeoutError('Kaggle task not observed COMPLETE by deadline')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','poll'])
    a=p.parse_args()
    k=prepare() if a.mode=='prepare' else f"{identity()}/tflex-v9-private-3d-gpu-probe"
    if a.mode=='poll':poll(k)
