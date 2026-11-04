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
    embeddings=[p for p in INPUT.rglob('motion_embeddings') if conf['model_kernel_slug'] in p.parts]
    if len(embeddings)!=1:raise FileNotFoundError('Private CPU-generated motion embeddings missing')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Real CUDA image-to-video requires an allocated GPU')
    pipe=load_pipeline(model)
    state={'success':False,'made':[],'errors':[],'reviewed':False,'published':False,'gpu_memory_pooled':False}
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
        try:
            for scene in range(conf['first_scene'],conf['first_scene']+conf['count']):
                for variant in range(1,conf.get('variants',1)+1):
                    image=image_root/f'images/scene_{scene:02d}.png'
                    dest=OUT/f'motion_outputs/{story}/motion/scene_{scene:02d}_{variant:02d}.mp4'
                    if dest.exists():continue
                    info=render_clip(pipe,embeddings[0]/f'{story}_{scene:02d}.safetensors',image,dest,
                        seed=int(story)*100000+scene*100+variant,width=conf.get('width',512),
                        height=conf.get('height',896),frames=conf.get('frames',121))
                    state['made'].append({'story':story,'scene':scene,'variant':variant,**info})
                    _json(OUT/'batch_motion_status.json',state)
        except Exception as exc:
            state['errors'].append({'story':story,'type':type(exc).__name__,'message':str(exc),
                                    'traceback':traceback.format_exc()})
        finally:
            folder=OUT/f'motion_outputs/{story}'
            if folder.exists():
                with zipfile.ZipFile(OUT/f'TwoWorlds_PRIVATE_{story}_motion.zip','w',zipfile.ZIP_STORED) as z:
                    for f in folder.rglob('*'):
                        if f.is_file():z.write(f,f'production_queue/{story}/'+str(f.relative_to(folder)))
            _json(OUT/'batch_motion_status.json',state)
    state['success']=len(state['errors'])==0 and len(state['made'])==len(conf['stories'])*conf['count']*conf.get('variants',1)
    _json(OUT/'batch_motion_status.json',state)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        (OUT/'batch_motion_failure.json').write_text(json.dumps({'success':False,'type':type(exc).__name__,
            'message':str(exc),'traceback':traceback.format_exc(),'reviewed':False,'published':False}))
        raise
