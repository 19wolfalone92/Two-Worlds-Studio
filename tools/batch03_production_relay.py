"""Private source handoff and bounded Kaggle production using existing secrets.

Public GitHub stores only reviewed code, public keys and authenticated ciphertext.
Account credentials never leave the GitHub runner or enter any returned artifact.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from studio.sealed_artifacts import generate_keys, seal, unseal

ROOT = Path(__file__).resolve().parents[1]
REQUEST_DIR = os.environ.get('TW_REQUEST_DIR', '.github/batch03_requests')
RELAY_BRANCH = os.environ.get('TW_RELAY_BRANCH', 'integration/story-glitch-batch03-20261104')
if REQUEST_DIR not in {'.github/batch03_requests', '.github/shortform04_requests', '.github/remaster02_requests'}:
    raise ValueError('Unsupported private request directory')
if RELAY_BRANCH not in {'integration/story-glitch-batch03-20261104', 'production/story-glitch-batch04-20261105', 'remaster/batch02-20261105'}:
    raise ValueError('Unsupported production branch')
TEMP = Path(os.environ.get('RUNNER_TEMP', '/tmp')) / ('tw-remaster02-relay' if 'remaster02' in REQUEST_DIR else 'tw-shortform04-relay' if 'shortform04' in REQUEST_DIR else 'tw-batch03-relay')


def request():
    doc = json.loads((ROOT / REQUEST_DIR / 'production.json').read_text())
    if (doc.get('schema') != 1 or doc.get('free_resources_only') is not True
            or doc.get('story') not in [str(i) for i in range(13, 19)]
            or doc.get('phase') not in {'seed_only', 'seed_image', 'image', 'image_batch', 'voice', 'voice_batch', 'voice_qa', 'motion_prepare', 'motion_batch', 'preflight', 'recover', 'status'}
            or not 1 <= doc.get('count', 0) <= 8):
        raise ValueError('Invalid explicitly bounded production request')
    import re
    if not re.fullmatch(r'[a-z0-9-]{6,64}', doc.get('nonce', '')):
        raise ValueError('Invalid request nonce')
    if not re.fullmatch(r'two-worlds-private-batch03-[a-z0-9-]{8,40}', doc.get('dataset_slug', '')):
        raise ValueError('Invalid private production dataset')
    if not re.fullmatch(r'[0-9a-f]{64}', doc.get('source_sha256', '')):
        raise ValueError('Missing immutable source checksum')
    if not 1 <= doc.get('first_scene',1) <= 8 or doc.get('first_scene',1)+doc['count']>9:
        raise ValueError('Scene range exceeds the approved eight-scene source')
    if 'stories' in doc and (not isinstance(doc['stories'],list) or not doc['stories']
            or len(set(doc['stories']))!=len(doc['stories'])
            or any(type(s) is not int or not 13<=s<=18 for s in doc['stories'])):
        raise ValueError('Invalid explicitly bounded story set')
    if doc['phase']=='motion_batch' and (not 1<=doc.get('variants',1)<=3
            or type(doc.get('first_variant',1)) is not int or not 1<=doc.get('first_variant',1)<=7
            or doc.get('vae_tile',512) not in [256,512]
            or doc.get('frames',121) not in range(49,258,8)
            or doc.get('width',512) not in [512,576,640,704]
            or doc.get('height',896) not in [896,1024,1152,1248]):
        raise ValueError('Invalid explicitly bounded motion request')
    if doc['phase'] == 'recover' and not re.fullmatch(
            r'wolf19loner92/(tw-batch03-[a-z0-9-]+|two-worlds-russian-(short|story)-gpu)', doc.get('kernel', '')):
        raise ValueError('Only this production kernel may be recovered')
    if doc['phase'] == 'status' and (not isinstance(doc.get('kernels'), list)
            or not 1 <= len(doc['kernels']) <= 4
            or any(not re.fullmatch(r'wolf19loner92/tw-batch03-[a-z0-9-]+', k)
                   for k in doc['kernels'])):
        raise ValueError('Only bounded existing production status queries are permitted')
    return doc


def prepare():
    conf = request()
    TEMP.mkdir(mode=0o700)
    if conf['phase'] in {'seed_image', 'seed_only'}:
        generate_keys(TEMP / 'input_private.pem', ROOT / 'dist/handshake/input_public.pem')
    else:
        (ROOT / 'dist/handshake').mkdir(parents=True, exist_ok=True)
        (ROOT / 'dist/handshake/no_input_needed.txt').write_text('Existing private dataset is reused.')


def command(args, *, timeout=180):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    # Raw provider responses and private kernel logs remain private.
    if result.returncode:
        diagnostic = result.stdout + result.stderr
        for name in ['KAGGLE_API_TOKEN', 'GH_READ_TOKEN']:
            secret = os.environ.get(name)
            if secret:
                diagnostic = diagnostic.replace(secret, '[REDACTED]')
        (TEMP / 'last_cli_error.txt').write_text(diagnostic)
        raise RuntimeError('Kaggle operation failed: ' + ' '.join(args[:3]))
    return result.stdout


def receive(conf):
    name = 'input_' + conf['nonce'] + '.twenc.b64'
    endpoint = ('https://api.github.com/repos/19wolfalone92/Two-Worlds-Studio/contents/'
                + REQUEST_DIR + '/' + name + '?ref=' +
                urllib.parse.quote(RELAY_BRANCH, safe=''))
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        req = urllib.request.Request(endpoint, headers={
            'Authorization': 'Bearer ' + os.environ['GH_READ_TOKEN'],
            'Accept': 'application/vnd.github.raw+json' if 'remaster02' in REQUEST_DIR else 'application/vnd.github+json', 'Cache-Control': 'no-cache'})
        try:
            if 'remaster02' in REQUEST_DIR:
                count=conf.get('input_part_count',1)
                if type(count) is not int or not 1<=count<=128:
                    raise ValueError('Invalid bounded encrypted source part count')
                parts=[]
                for index in range(count):
                    part_name='input_'+conf['nonce']+f'.part{index:04d}.twenc.b64'
                    part_req=urllib.request.Request(endpoint.replace(name,part_name),headers=dict(req.headers))
                    with urllib.request.urlopen(part_req,timeout=30) as response:
                        parts.append(base64.b64decode(response.read(),validate=True))
                encrypted=b''.join(parts)
            else:
                with urllib.request.urlopen(req, timeout=30) as response:
                    payload = json.load(response)
                    encrypted = base64.b64decode(base64.b64decode(payload['content']))
            sealed = TEMP / 'input.twenc'
            sealed.write_bytes(encrypted)
            source = TEMP / 'batch03_source.zip'
            unseal(sealed, TEMP / 'input_private.pem', source,
                   {'purpose': 'batch03-private-source', 'nonce': conf['nonce']})
            if hashlib.sha256(source.read_bytes()).hexdigest() != conf['source_sha256']:
                raise ValueError('Private source checksum mismatch')
            return source
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise RuntimeError('Private source transport returned HTTP ' + str(exc.code)) from None
        time.sleep(15)
    raise TimeoutError('Private source handoff did not arrive within 15 minutes')


def seal_return(archive, recipient, output, nonce, *, part_bytes=80*1024*1024):
    """Authenticate bounded parts instead of losing a successful large batch."""
    output.mkdir(parents=True,exist_ok=True)
    context={'purpose':'batch03-production','nonce':nonce}
    if archive.stat().st_size <= part_bytes:
        seal(archive,recipient,output/'production_result.twenc',context)
        return
    if archive.stat().st_size>2*1024**3:
        raise ValueError('Return exceeds the explicit 2GB artifact budget')
    count=(archive.stat().st_size+part_bytes-1)//part_bytes
    with archive.open('rb') as stream:
        archive_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    parts=[]
    with archive.open('rb') as stream:
        for index in range(count):
            raw=TEMP/f'return_part_{index:04d}.bin'
            raw.write_bytes(stream.read(part_bytes))
            name=f'production_result.part{index:04d}.twenc'
            ctx={**context,'part':index,'count':count,'archive_sha256':archive_hash}
            seal(raw,recipient,output/name,ctx)
            raw.unlink()
            parts.append({'index':index,'file':name})
    (output/'return_parts.json').write_text(json.dumps({'schema':1,'nonce':nonce,
        'archive_sha256':archive_hash,'count':count,'parts':parts},indent=2))


def produce():
    conf = request()
    summary = {'nonce': conf['nonce'], 'phase': conf['phase'], 'story': conf['story'],
               'paid_services': False, 'published': False}
    result_dir = TEMP / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        quota = command(['kaggle', 'quota'])
        if conf['phase'] == 'status':
            summary['quota'] = quota
            summary['kernel_statuses'] = {k: command(['kaggle', 'kernels', 'status', k])
                                          for k in conf['kernels']}
            if conf.get('query_progress') is True:
                summary['progress_queries'] = {}
                for index, kernel in enumerate(conf['kernels']):
                    try:
                        command(['kaggle', 'kernels', 'output', kernel,
                                 '-p', str(result_dir / f'progress_{index}'), '--force',
                                 '--file-pattern', r'(batch_(image|voice)_status\.json|TwoWorlds_PRIVATE_.*\.zip)$'],
                                timeout=180)
                        summary['progress_queries'][kernel] = 'AVAILABLE'
                    except Exception as exc:
                        summary['progress_queries'][kernel] = type(exc).__name__
            summary['gpu_requested'] = False
            return
        dataset = 'wolf19loner92/' + conf['dataset_slug']
        if conf['phase'] == 'recover':
            from studio.kaggle_api import poll_kernel
            summary['kernel'] = conf['kernel']
            try:
                summary['kernel_status'] = poll_kernel(conf['kernel'], timeout_seconds=3500,
                                                       interval_seconds=45)
            except Exception as exc:
                summary['kernel_error'] = type(exc).__name__ + ': ' + str(exc)
            command(['kaggle', 'kernels', 'output', conf['kernel'], '-p', str(result_dir),
                     '--force', '--file-pattern', r'(TwoWorlds_PRIVATE_.*\.zip|two_worlds_kaggle_receipt\.json|batch_(voice|image|motion)_.*\.json)$'],
                    timeout=300)
            return
        if conf['phase'] in {'seed_image', 'seed_only'}:
            source = receive(conf)
            upload = TEMP / 'source_upload'
            upload.mkdir()
            (upload / 'batch03_source.zip').write_bytes(source.read_bytes())
            meta = {'id': dataset, 'title': 'Two Worlds private Batch03 production source',
                    'licenses': [{'name': 'other'}]}
            (upload / 'dataset-metadata.json').write_text(json.dumps(meta))
            # Kaggle create defaults to private. No public flag is accepted here.
            command(['kaggle', 'datasets', 'create', '-p', str(upload)], timeout=300)
            summary['private_dataset_created'] = True
            # Verify the upload is queryable before requesting any GPU.
            ready = False
            for _ in range(20):
                try:
                    command(['kaggle', 'datasets', 'files', dataset])
                    ready = True
                    break
                except RuntimeError:
                    time.sleep(10)
            if not ready:
                raise RuntimeError('Private source dataset did not become available')
            if conf['phase'] == 'seed_only':
                summary['gpu_requested'] = False
                return
        from studio.kaggle_shorts import build_kernel
        from studio.kaggle_api import poll_kernel
        phase = 'image' if conf['phase'] == 'seed_image' else conf['phase']
        build_phase = ('voice' if phase=='voice_batch' else 'image' if phase in {'image_batch','motion_batch'}
                       else 'preflight' if phase in {'motion_prepare','voice_qa'} else phase)
        cached = conf.get('model_kernel')
        repair_cache = bool(cached and conf.get('repair_cached_model'))
        slug = 'tw-batch03-' + conf['nonce']
        kernel_dir = TEMP / 'kernel'
        prepared = build_kernel(ROOT, kernel_dir, username='wolf19loner92',
            dataset=dataset, phase=build_phase, story=conf['story'],
            first_scene=conf.get('first_scene', 1), max_scenes=conf['count'],
            budget_s=3000, slug=slug,
            source_path='/kaggle/input/' + conf['dataset_slug'] + '/batch03_source.zip',
            model_path=('/kaggle/input/' + cached.split('/')[1] + '/private_models' if cached and not repair_cache else
                        '/kaggle/working/private_models' if phase in {'image', 'voice'}
                        else '/kaggle/input/no-models-required'),
            model_kernel=cached,
            recovery_kernel=conf.get('recovery_kernel'), recovery_path=conf.get('recovery_path'),
            speaker=conf.get('speaker', 'Ryan' if conf['story'] in {'14', '17'} else 'Serena'),
            install_deps=True, bootstrap=(build_phase in {'image', 'voice'} and (not cached or repair_cache)))
        if phase == 'voice_batch':
            if not cached:
                raise ValueError('The full batch requires the already validated shared Qwen cache')
            from studio.kaggle_shorts import build_source
            batch_conf = {'source_path':'/kaggle/input/'+conf['dataset_slug']+'/batch03_source.zip',
                          'dataset_slug':conf['dataset_slug'],
                          'model_root':'/kaggle/input/'+cached.split('/')[1]+'/private_models',
                          'model_kernel_slug':cached.split('/')[1]}
            if 'tasks' in conf:
                from tools.kaggle_voice_batch_entry import voice_tasks
                batch_conf['tasks'] = voice_tasks(conf)
            if 'speakers' in conf:
                from studio.offline_media import SPEAKERS
                if any(str(k) not in {str(i) for i in range(13,19)} or v not in SPEAKERS
                       for k,v in conf['speakers'].items()):
                    raise ValueError('Invalid bounded narrator cast')
                batch_conf['speakers'] = conf['speakers']
            template = (ROOT/'tools/kaggle_voice_batch_entry.py').read_text()
            rendered = template.replace('__TW_VOICE_BATCH_CONFIG__',base64.b64encode(json.dumps(batch_conf).encode()).decode())
            rendered = rendered.replace('__TW_VOICE_BATCH_SOURCE__',base64.b64encode(build_source(ROOT)).decode())
            (kernel_dir/'kernel_main.py').write_text(rendered)
        if phase == 'image_batch':
            if not cached:
                raise ValueError('Image batch requires the already validated shared FLUX cache')
            from studio.kaggle_shorts import build_source
            batch_conf = {'source_path':'/kaggle/input/'+conf['dataset_slug']+'/batch03_source.zip',
                          'dataset_slug':conf['dataset_slug'],
                          'model_root':'/kaggle/input/'+cached.split('/')[1]+'/private_models',
                          'model_kernel_slug':cached.split('/')[1],
                          'stories':conf.get('stories', list(range(13,19))),
                          'first_scene':conf.get('first_scene',1),'count':conf['count'],
                          'width':conf.get('width',864),'height':conf.get('height',1536)}
            template = (ROOT/'tools/kaggle_image_batch_entry.py').read_text()
            rendered = template.replace('__TW_IMAGE_BATCH_CONFIG__',base64.b64encode(json.dumps(batch_conf).encode()).decode())
            rendered = rendered.replace('__TW_IMAGE_BATCH_SOURCE__',base64.b64encode(build_source(ROOT)).decode())
            (kernel_dir/'kernel_main.py').write_text(rendered)
        if phase in {'motion_prepare','motion_batch'}:
            from studio.kaggle_shorts import build_source
            if phase=='motion_batch' and not (cached and conf.get('image_kernel')):
                raise ValueError('Motion inference requires validated model and image output mounts')
            batch_conf={'phase':phase,'dataset_slug':conf['dataset_slug'],
                        'model_kernel_slug':cached.split('/')[1] if cached else None,
                        'image_kernel_slug':conf.get('image_kernel','/').split('/')[1],
                        'image_fallback_slug':conf.get('recovery_kernel','/').split('/')[1],
                        'memory_mode':conf.get('motion_memory_mode','cpu_offload'),
                        'vae_tile':conf.get('vae_tile',512),'first_variant':conf.get('first_variant',1),
                        'stories':conf.get('stories',[13]),'first_scene':conf.get('first_scene',1),
                        'count':conf['count'],'variants':conf.get('variants',1),
                        'frames':conf.get('frames',121),'width':conf.get('width',512),'height':conf.get('height',896)}
            template=(ROOT/'tools/kaggle_motion_entry.py').read_text()
            rendered=template.replace('__TW_MOTION_CONFIG__',base64.b64encode(json.dumps(batch_conf).encode()).decode())
            rendered=rendered.replace('__TW_MOTION_SOURCE__',base64.b64encode(build_source(ROOT)).decode())
            (kernel_dir/'kernel_main.py').write_text(rendered)
        if phase=='voice_qa':
            if not conf.get('recovery_kernel'):raise ValueError('Speech audit requires the actual full voice outputs')
            if conf.get('qa_engine','wav2vec2') not in {'wav2vec2','whisper_cpu'}:raise ValueError('Unknown CPU speech audit engine')
            if not cached and conf.get('qa_engine','wav2vec2')=='wav2vec2':raise ValueError('CTC speech audit requires validated cached weights')
            from studio.kaggle_shorts import build_source
            batch_conf={'model_kernel_slug':cached.split('/')[1] if cached else None,
                        'voice_kernel_slug':conf['recovery_kernel'].split('/')[1],
                        'qa_scenes':conf.get('qa_scenes')}
            template=(ROOT/('tools/kaggle_voice_whisper_entry.py' if conf.get('qa_engine')=='whisper_cpu' else 'tools/kaggle_voice_qa_entry.py')).read_text()
            rendered=template.replace('__TW_VOICE_QA_CONFIG__',base64.b64encode(json.dumps(batch_conf).encode()).decode())
            rendered=rendered.replace('__TW_VOICE_QA_SOURCE__',base64.b64encode(build_source(ROOT)).decode())
            (kernel_dir/'kernel_main.py').write_text(rendered)
        metadata_path = kernel_dir / 'kernel-metadata.json'
        metadata = json.loads(metadata_path.read_text())
        if conf.get('image_kernel'):
            if not re.fullmatch(r'wolf19loner92/tw-batch03-[a-z0-9-]+',conf['image_kernel']):
                raise ValueError('Only validated private Batch03 output may supply motion input')
            metadata['kernel_sources']=list(dict.fromkeys(metadata['kernel_sources']+[conf['image_kernel']]))
        # Kaggle can derive a new kernel's URL suffix from its title.
        metadata['title'] = slug.replace('-', ' ')
        metadata_path.write_text(json.dumps(metadata, indent=2))
        summary['kernel'] = prepared['kernel']
        args = ['kaggle', 'kernels', 'push', '-p', str(kernel_dir), '--timeout',
                '10800' if phase in {'voice_batch','image_batch','voice_qa','motion_prepare','motion_batch'} else '3600']
        if phase in {'image', 'image_batch', 'voice', 'voice_batch','motion_batch'}:
            args += ['--accelerator', 'NvidiaTeslaT4']
        pushed = command(args)
        (result_dir / 'submission_response.txt').write_text(pushed)
        if 'Kernel push error:' in pushed:
            raise RuntimeError('Kaggle rejected this session request; no worker was submitted')
        urls = re.findall(r'https://www\.kaggle\.com/code/(wolf19loner92/[a-z0-9-]+)', pushed)
        if urls:
            prepared['kernel'] = urls[-1]
            summary['kernel'] = prepared['kernel']
        summary['submitted'] = True
        try:
            summary['kernel_status'] = poll_kernel(prepared['kernel'],
                timeout_seconds=9600 if phase in {'voice_batch','image_batch','voice_qa','motion_prepare','motion_batch'} else 3500, interval_seconds=45)
        except Exception as exc:
            summary['kernel_error'] = type(exc).__name__ + ': ' + str(exc)
        command(['kaggle', 'kernels', 'output', prepared['kernel'], '-p', str(result_dir),
                 '--force', '--file-pattern', r'(TwoWorlds_PRIVATE_.*\.zip|two_worlds_kaggle_receipt\.json|batch_(voice|image|motion)_.*\.json)$'],
                timeout=300)
        summary['downloaded_files'] = len([p for p in result_dir.rglob('*') if p.is_file()])
    except Exception as exc:
        summary['error_type'] = type(exc).__name__
        summary['error'] = str(exc)
    finally:
        (result_dir / 'relay_summary.json').write_text(json.dumps(summary, indent=2))
        if (TEMP / 'last_cli_error.txt').exists():
            (result_dir / 'private_cli_diagnostic.txt').write_bytes((TEMP / 'last_cli_error.txt').read_bytes())
        archive = TEMP / 'private_return.zip'
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for path in result_dir.rglob('*'):
                if path.is_file():
                    z.write(path, str(path.relative_to(result_dir)))
        # NEVER include runner input private key, account environment, or credentials.
        recipient_dir = REQUEST_DIR if 'remaster02' in REQUEST_DIR else '.github/batch03_requests'
        seal_return(archive,ROOT/recipient_dir/'recipient_public.pem',
                    ROOT/'dist/private_batch03',conf['nonce'])
        print(json.dumps({k: summary[k] for k in ['nonce', 'phase', 'story', 'paid_services', 'published']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'produce'])
    getattr(__import__(__name__, fromlist=['']), parser.parse_args().action)()
