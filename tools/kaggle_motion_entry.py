"""Private LTX CPU staging and bounded real image-to-video passes."""
from __future__ import annotations
import base64
import io
import json
import os
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

CONFIG_B64='__TW_MOTION_CONFIG__'
SOURCE_B64='__TW_MOTION_SOURCE__'
OUT=Path('/kaggle/working')
INPUT=Path('/kaggle/input')
BASE=OUT/'tw_motion_modules'


def bounded_tasks(conf):
    return [(str(story),scene,variant) for story in conf['stories']
            for scene in range(conf['first_scene'],conf['first_scene']+conf['count'])
            for variant in range(1,conf.get('variants',1)+1)]


def worker(config):
    """One independent model per visible card; VRAM is never pooled."""
    sys.path.insert(0,str(BASE))
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(2)
    from studio.neural_motion import load_pipeline,render_clip,_json
    pipe=load_pipeline(Path(config['model']))
    state={'made':[],'errors':[],'gpu_memory_pooled':False,'reviewed':False}
    status=Path(config['status'])
    for story,scene,variant in config['tasks']:
        dest=OUT/f'motion_outputs/{story}/motion/scene_{scene:02d}_{variant:02d}.mp4'
        try:
            if dest.exists() and dest.with_suffix('.receipt.json').exists():
                info=json.loads(dest.with_suffix('.receipt.json').read_text())
                from studio.neural_motion import _digest
                if info['sha256']!=_digest(dest):raise ValueError('Recovered motion checksum mismatch')
                info['reused']=True
            else:
                info=render_clip(pipe,Path(config['embeddings'])/f'{story}_{scene:02d}.safetensors',
                    Path(config['images'][story])/f'images/scene_{scene:02d}.png',dest,
                    seed=int(story)*100000+scene*100+variant,width=config['width'],
                    height=config['height'],frames=config['frames'])
            state['made'].append({'story':story,'scene':scene,'variant':variant,**info})
        except Exception as exc:
            state['errors'].append({'story':story,'scene':scene,'variant':variant,
                'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
            if isinstance(exc,torch.cuda.OutOfMemoryError):
                import gc
                gc.collect();torch.cuda.empty_cache()
        _json(status,state)
    return 0


def main():
    conf=json.loads(base64.b64decode(CONFIG_B64))
    BASE.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as z:
        for item in z.infolist():
            p=Path(item.filename)
            if p.is_absolute() or '..' in p.parts or p.parts[0]!='studio':raise ValueError('Unsafe source bundle')
        z.extractall(BASE)
    sys.path.insert(0,str(BASE))
    os.environ['HF_HUB_DISABLE_XET']='1'
    subprocess.run([sys.executable,'-m','pip','install','-q','diffusers','transformers',
                    'accelerate','safetensors','sentencepiece','imageio-ffmpeg'],check=True,timeout=1200)
    from studio.neural_motion import stage_weights,encode_directions,load_pipeline,render_clip,_json
    projects=[p for p in INPUT.rglob('project.json') if p.parent.name=='13'
              and p.parent.parent.name=='production_queue' and conf['dataset_slug'] in p.parts]
    if len(projects)!=1:raise FileNotFoundError('Original private source missing or ambiguous')
    source=projects[0].parent.parent.parent
    if conf['phase']=='motion_prepare':
        home=OUT/'private_models/motion'
        stage=stage_weights(home)
        embeddings=encode_directions(source,home,OUT/'motion_embeddings')
        _json(OUT/'batch_motion_stage.json',{'success':True,'stage':stage,'embeddings':embeddings,
                                             'gpu_requested':False,'reviewed':False,'published':False})
        return
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    models=[p.parent for p in INPUT.rglob('stage_receipt.json') if p.parent.name=='motion'
            and conf['model_kernel_slug'] in p.parts]
    if len(models)!=1:raise FileNotFoundError('Validated private LTX model cache missing or ambiguous')
    model=models[0]
    # The GPU pass needs only transformer+VAE. Leave the large CPU text encoder
    # attached; its private embeddings have already been computed once.
    from studio.model_cache import stage_cached_tree
    import tempfile,shutil
    local_model=Path(tempfile.mkdtemp(prefix='tw-ltx-components-'))
    for part in ['transformer','vae']:
        copied,receipt=stage_cached_tree(model/part)
        copied.rename(local_model/part)
    model=local_model
    embeddings=[p for p in INPUT.rglob('motion_embeddings') if conf['model_kernel_slug'] in p.parts]
    if len(embeddings)!=1:raise FileNotFoundError('Private CPU-generated motion embeddings missing')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Real CUDA image-to-video requires an allocated GPU')
    torch.set_num_threads(2)
    state={'success':False,'made':[],'errors':[],'reviewed':False,'published':False,'gpu_memory_pooled':False}
    image_roots={}
    for story in conf['stories']:
        story=str(story)
        projects=[p for p in INPUT.rglob('project.json') if p.parent.name==story
                  and conf['image_kernel_slug'] in p.parts and p.parent.parent.name=='production_queue']
        image_root=projects[0].parent if len(projects)==1 else None
        archive=[p for p in INPUT.rglob(f'TwoWorlds_PRIVATE_{story}_image.zip') if conf['image_kernel_slug'] in p.parts]
        if image_root is None and len(archive)==1:
            temp=OUT/f'image_inputs/{story}'
            from studio.kaggle_shorts_worker import _safe_extract
            with zipfile.ZipFile(archive[0]) as z:_safe_extract(z,temp)
            image_root=temp/'production_queue'/story
        if image_root is None:
            state['errors'].append({'story':story,'type':'IMAGE_SOURCE_MISSING'});continue
        image_roots[story]=str(image_root)
    tasks=[task for task in bounded_tasks(conf) if task[0] in image_roots]
    devices=min(2,torch.cuda.device_count())
    jobs=[]
    for card in range(devices):
        settings={'model':str(model),'embeddings':str(embeddings[0]),'images':image_roots,
            'tasks':tasks[card::devices],'width':conf.get('width',512),'height':conf.get('height',896),
            'frames':conf.get('frames',81),'status':str(OUT/f'motion_worker_{card}.json')}
        path=OUT/f'motion_worker_{card}_config.json';_json(path,settings)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(card),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        jobs.append(subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(path)],env=env))
    state['independent_gpu_workers']=devices
    _json(OUT/'batch_motion_status.json',state)
    for card,job in enumerate(jobs):
        code=job.wait()
        status=OUT/f'motion_worker_{card}.json'
        if status.exists():
            result=json.loads(status.read_text());state['made']+=result['made'];state['errors']+=result['errors']
        if code:state['errors'].append({'worker':card,'type':'WORKER_EXIT','code':code})
    for story in conf['stories']:
        folder=OUT/f'motion_outputs/{story}'
        if folder.exists():
            with zipfile.ZipFile(OUT/f'TwoWorlds_PRIVATE_{story}_motion.zip','w',zipfile.ZIP_STORED) as z:
                for f in folder.rglob('*'):
                    if f.is_file():z.write(f,f'production_queue/{story}/'+str(f.relative_to(folder)))
    state['success']=len(state['errors'])==0 and len(state['made'])==len(conf['stories'])*conf['count']*conf.get('variants',1)
    _json(OUT/'batch_motion_status.json',state)


if __name__=='__main__':
    try:
        if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(json.loads(Path(sys.argv[2]).read_text()))
        else:main()
    except Exception as exc:
        (OUT/'batch_motion_failure.json').write_text(json.dumps({'success':False,'type':type(exc).__name__,
            'message':str(exc),'traceback':traceback.format_exc(),'reviewed':False,'published':False}))
        raise
