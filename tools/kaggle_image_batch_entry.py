"""One cached FLUX load for a private, resumable, cast-controlled image batch.

The public entrypoint contains no story text or character descriptions. Those
are read only from the authorized private Dataset. CUDA receipts are factual;
every output still requires visual review.
"""
from __future__ import annotations
import base64
import gc
import io
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

CONFIG_B64 = '__TW_IMAGE_BATCH_CONFIG__'
SOURCE_B64 = '__TW_IMAGE_BATCH_SOURCE__'
OUT = Path('/kaggle/working')
INPUT = Path('/kaggle/input')
BASE = OUT/'tw_image_modules'


def resolve_source(hint: Path, slug: str, input_root: Path = INPUT) -> Path:
    if hint.is_file():
        return hint
    files = [p for p in input_root.rglob(hint.name) if slug in p.parts and p.is_file()]
    if len(files) == 1:
        return files[0]
    projects = [p for p in input_root.rglob('project.json')
                if p.parent.name == '13' and p.parent.parent.name == 'production_queue'
                and slug in p.parts]
    if len(projects) == 1:
        return projects[0].parent.parent.parent
    raise FileNotFoundError('Private source missing or ambiguous')


def scene_prompt(director: dict, shot: dict, repair: dict | None = None) -> tuple[str, list[str]]:
    repair=repair or {}
    cast = repair.get('cast',director['scene_cast'][str(shot['scene'])])
    if len(cast) > 4 or any(p not in director['people'] for p in cast):
        raise ValueError('Unknown character or too many identity references')
    descriptions = ' '.join(
        f"Reference image {i+1} identifies {director['people'][key]['name']}: "
        f"{director['people'][key]['description']}"
        for i, key in enumerate(cast))
    # Put the actual shot before generic portrait/continuity language. On a
    # long multi-cast prompt, the old ordering could bury object-only staging
    # and produce an irrelevant portrait instead of the requested story prop.
    if director.get('shot_first_prompt'):
        staging=repair.get('staging',shot['visual_prompt_ru'])
        prompt=('ONE vertical photorealistic film frame. EXACT SHOT: '+staging+' '
                +director.get('scene_notes',{}).get(str(shot['scene']),'')+' ')
        if cast:
            prompt+=('Use the reference images only to identify the listed people. '
                     'Honor the framing: never widen a hands-only or object-detail shot into a face portrait. '
                     +descriptions+' ')
        else:
            prompt+='This is strictly an object or environment shot. ZERO PEOPLE, ZERO FACES, ZERO HUMAN BODY PARTS. '
        prompt+=('Lighting and color: '+director['palette']+'. Realistic anatomy, real materials, '
                 'natural skin and practical cinematic lighting. One coherent image, no montage, '
                 'extra people, captions, logos or watermarks. Leave requested blank paper blank.')
        return prompt,cast
    prompt = (
        'Create ONE cinematic live-action movie frame, vertical composition. '
        + ('Preserve each referenced face, age and hair exactly; follow the scene instructions for clothing. '
           if repair.get('change_clothing') else 'Preserve each referenced face, age, hair and clothing exactly; ')
        + 'change the staging to the specified scene. Natural expressive faces, '
        'physically credible anatomy and hands, motivated practical lighting, '
        'real skin texture, coherent reflections and shadows, restrained film grain, '
        'professional 35mm cinematography, not an illustration or collage. '
        + descriptions + ' Scene: ' + repair.get('staging',shot['visual_prompt_ru']) + ' '
        + director.get('scene_notes', {}).get(str(shot['scene']), '')
        + ' Color and lighting continuity: ' + director['palette'] + '. '
        + ('Only the listed characters may be in the frame. ' if cast else
           'No people or human silhouettes in this frame. ')
        + 'Leave documents and screens without invented readable lettering. '
          'No logos, captions, watermarks, extra people, duplicated faces or bodies.')
    return prompt, cast


def main():
    conf = json.loads(base64.b64decode(CONFIG_B64))
    stories = [str(i) for i in conf['stories']]
    if len(set(stories)) != len(stories) or any(s not in {str(i) for i in range(13,19)} for s in stories):
        raise ValueError('Only the six approved source stories can be generated')
    BASE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as z:
        for item in z.infolist():
            p = Path(item.filename)
            if p.is_absolute() or '..' in p.parts or p.parts[0] != 'studio':
                raise ValueError('Unsafe module bundle')
        z.extractall(BASE)
    sys.path.insert(0,str(BASE))
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    subprocess.run([sys.executable,'-m','pip','install','-q',
                    'diffusers','transformers','accelerate','safetensors'],check=True,timeout=1200)
    import torch
    from PIL import Image
    from diffusers import Flux2KleinPipeline
    from studio.offline_media import real_image, configure_image_memory, digest
    from studio.kaggle_shorts_worker import prepare_input, export_private_assets
    from studio.autonomous_factory import atomic_json, inventory
    from studio.model_cache import stage_cached_tree
    if not torch.cuda.is_available():
        raise RuntimeError('Actual image inference requires a real CUDA accelerator')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    model_root = Path(conf['model_root'])
    if not (model_root/'image/model_index.json').is_file():
        found = [p.parent.parent for p in INPUT.rglob('model_index.json')
                 if p.parent.name == 'image' and conf['model_kernel_slug'] in p.parts]
        if len(found) != 1:
            raise FileNotFoundError('Validated FLUX component cache missing or ambiguous')
        model_root = found[0]
    source = resolve_source(Path(conf['source_path']),conf['dataset_slug'])
    width,height = int(conf['width']),int(conf['height'])
    if width % 16 or height % 16 or not (512 <= width <= 1088 and 768 <= height <= 1920):
        raise ValueError('Invalid bounded portrait resolution')
    state = {'schema':1,'status':'RUNNING','model':'black-forest-labs/FLUX.2-klein-4B',
             'license':'Apache-2.0','gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
             'memory_pooled':False,'stories':{},'reviewed':False,'published':False,'paid_services':False}
    started = time.monotonic()
    def checkpoint():
        state['elapsed_s'] = round(time.monotonic()-started,2)
        atomic_json(OUT/'batch_image_status.json',state)
    local_image,stage_receipt=stage_cached_tree(model_root/'image')
    state['model_staging']=stage_receipt
    # Component placement uses the two separate cards with explicit per-card
    # limits. This never presents 16+16GB as one 32GB allocation.
    try:
        if torch.cuda.device_count() < 2:
            raise ValueError('Use the tested single-card offload path')
        pipe = Flux2KleinPipeline.from_pretrained(str(local_image),
                    dtype=torch.float16,local_files_only=True,
                    device_map='balanced',max_memory={0:'13GiB',1:'13GiB','cpu':'20GiB'})
        if str(pipe.hf_device_map.get('transformer','')).startswith('cpu'):
            raise ValueError('Do not silently run the diffusion transformer on CPU')
        if next(pipe.transformer.parameters()).dtype != torch.float16:
            raise ValueError('Unexpected FLUX inference dtype')
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        state['placement'] = {str(k):str(v) for k,v in pipe.hf_device_map.items()}
    except (ValueError,NotImplementedError,torch.cuda.OutOfMemoryError) as exc:
        state['placement_fallback'] = type(exc).__name__
        if 'pipe' in locals():
            del pipe
        gc.collect()
        torch.cuda.empty_cache()
        pipe = Flux2KleinPipeline.from_pretrained(str(local_image),
                    torch_dtype=torch.float16,local_files_only=True)
        state['placement'] = configure_image_memory(pipe,torch.cuda.get_device_properties(0).total_memory/1024**3)
    checkpoint()
    for story in stories:
        if time.monotonic()-started > 9000:
            state['status'] = 'TIME_BUDGET_REACHED'
            break
        work = OUT/'private_image_projects'/story
        recovery = [p for p in INPUT.rglob(f'TwoWorlds_PRIVATE_{story}_image.zip')
                    if conf['model_kernel_slug'] not in p.parts]
        prepare_input(source,work,story,recovery=recovery[0] if len(recovery)==1 else None)
        director = json.loads((work/'image_director.json').read_text())
        repairs_path=work/'image_repairs.json'
        repairs=json.loads(repairs_path.read_text()).get('scenes',{}) if repairs_path.exists() else {}
        shots = json.loads((work/'shot_plan.json').read_text())
        item = {'status':'RUNNING','made':[],'skipped':[],'references':{},'errors':[]}
        state['stories'][story] = item
        checkpoint()
        try:
            refs = work/'images/refs'
            refs.mkdir(parents=True,exist_ok=True)
            bootstrap = director.get('identity_bootstrap',{})
            # A rejected staging frame may supply face crops, but is NEVER
            # imported as a scene. Each crop becomes a separate cast reference.
            old = list(INPUT.rglob(bootstrap.get('archive','__absent__')))
            if len(old)==1:
                with zipfile.ZipFile(old[0]) as z:
                    frame = Image.open(io.BytesIO(z.read(bootstrap['image']))).convert('RGB')
                for key,box in bootstrap['crops'].items():
                    target = refs/(key+'.png')
                    if not target.exists():
                        frame.crop(tuple(box)).save(target)
            for index,(key,person) in enumerate(director['people'].items()):
                target = refs/(key+'.png')
                if not target.is_file():
                    prompt = ('ONE real person, an individual cinema casting portrait. '
                        'Chest-up three-quarter view, soft gray background, soft natural daylight, '
                        'true skin texture, understated human expression, sharp natural eyes, '
                        'relaxed shoulders, 85mm lens, photorealistic live-action photography. '
                        + person['description'] + ' No other people, text, logos, watermarks or collage.')
                    info = real_image(prompt,target,model_root,pipe=pipe,seed=int(story)*1000+index,
                                      width=768,height=1024)
                    item['references'][key] = info
                    checkpoint()
                else:
                    item['references'][key] = {'sha256':digest(target),'reused':True}
            selected = [s for s in shots if conf['first_scene'] <= s['scene'] < conf['first_scene']+conf['count']]
            for shot in selected:
                if time.monotonic()-started > 9000:
                    item['errors'].append({'type':'TIME_BUDGET_REACHED'})
                    break
                target = work/f"images/scene_{shot['scene']:02d}.png"
                repair=repairs.get(str(shot['scene']),{})
                if target.is_file() and repair and digest(target)==repair.get('rejected_sha256'):
                    # Keep the rejected original in a separate private checkpoint.
                    rejected=OUT/f"rejected_images/{story}/{target.name}"
                    rejected.parent.mkdir(parents=True,exist_ok=True)
                    target.rename(rejected)
                if target.is_file():
                    item['skipped'].append(shot['scene'])
                    continue
                prompt,cast = scene_prompt(director,shot,repair)
                try:
                    info = real_image(prompt,target,model_root,pipe=pipe,
                        seed=int(story)*10000+shot['scene']+int(repair.get('seed_offset',0)),width=width,height=height,
                        references=[refs/(key+'.png') for key in cast])
                    info['scene'] = shot['scene']
                    item['made'].append(info)
                    doc = json.loads((work/'project.json').read_text())
                    next(s for s in doc['scenes'] if s['id']==shot['scene'])['image'] = str(target.relative_to(work))
                    atomic_json(work/'project.json',doc)
                    atomic_json(target.with_suffix('.receipt.json'),info)
                except torch.cuda.OutOfMemoryError:
                    # Do not silently reduce resolution or invent a substitute.
                    gc.collect();torch.cuda.empty_cache()
                    item['errors'].append({'scene':shot['scene'],'type':'CUDA_OUT_OF_MEMORY'})
                checkpoint()
            item['status'] = 'COMPLETE' if len(item['made'])+len(item['skipped'])==len(selected) and not item['errors'] else 'PARTIAL'
        except Exception as exc:
            item['status'] = 'PARTIAL_OR_FAILED'
            item['errors'].append({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        finally:
            inventory(work)
            bundle = export_private_assets(work,OUT/f'TwoWorlds_PRIVATE_{story}_image.zip',item)
            item['asset_bundle'] = bundle
            checkpoint()
    state['status'] = 'COMPLETE' if len(state['stories'])==len(stories) and all(s['status']=='COMPLETE' for s in state['stories'].values()) else 'PARTIAL'
    checkpoint()


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        (OUT/'batch_image_failure.json').write_text(json.dumps({'success':False,
            'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
            'reviewed':False,'published':False}))
        raise
