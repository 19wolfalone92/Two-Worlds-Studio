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
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from studio.sealed_artifacts import generate_keys, seal, unseal

ROOT = Path(__file__).resolve().parents[1]
TEMP = Path(os.environ.get('RUNNER_TEMP', '/tmp')) / 'tw-batch03-relay'


def request():
    doc = json.loads((ROOT / '.github/batch03_requests/production.json').read_text())
    if (doc.get('schema') != 1 or doc.get('free_resources_only') is not True
            or doc.get('story') not in [str(i) for i in range(13, 19)]
            or doc.get('phase') not in {'seed_image', 'image', 'voice', 'voice_batch', 'preflight', 'recover'}
            or not 1 <= doc.get('count', 0) <= 8):
        raise ValueError('Invalid explicitly bounded production request')
    import re
    if not re.fullmatch(r'[a-z0-9-]{6,64}', doc.get('nonce', '')):
        raise ValueError('Invalid request nonce')
    if not re.fullmatch(r'two-worlds-private-batch03-[a-z0-9-]{8,40}', doc.get('dataset_slug', '')):
        raise ValueError('Invalid private production dataset')
    if not re.fullmatch(r'[0-9a-f]{64}', doc.get('source_sha256', '')):
        raise ValueError('Missing immutable source checksum')
    if doc['phase'] == 'recover' and not re.fullmatch(
            r'wolf19loner92/(tw-batch03-[a-z0-9-]+|two-worlds-russian-(short|story)-gpu)', doc.get('kernel', '')):
        raise ValueError('Only this production kernel may be recovered')
    return doc


def prepare():
    conf = request()
    TEMP.mkdir(mode=0o700)
    if conf['phase'] == 'seed_image':
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
                '.github/batch03_requests/' + name + '?ref=' +
                urllib.parse.quote('integration/story-glitch-batch03-20261104', safe=''))
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        req = urllib.request.Request(endpoint, headers={
            'Authorization': 'Bearer ' + os.environ['GH_READ_TOKEN'],
            'Accept': 'application/vnd.github+json', 'Cache-Control': 'no-cache'})
        try:
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


def produce():
    conf = request()
    summary = {'nonce': conf['nonce'], 'phase': conf['phase'], 'story': conf['story'],
               'paid_services': False, 'published': False}
    result_dir = TEMP / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        command(['kaggle', 'quota'])
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
                     '--force', '--file-pattern', r'(TwoWorlds_PRIVATE_.*\.zip|two_worlds_kaggle_receipt\.json)$'],
                    timeout=300)
            return
        if conf['phase'] == 'seed_image':
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
        from studio.kaggle_shorts import build_kernel
        from studio.kaggle_api import poll_kernel
        phase = 'image' if conf['phase'] == 'seed_image' else conf['phase']
        build_phase = 'voice' if phase == 'voice_batch' else phase
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
                          'model_root':'/kaggle/input/'+cached.split('/')[1]+'/private_models',
                          'model_kernel_slug':cached.split('/')[1]}
            template = (ROOT/'tools/kaggle_voice_batch_entry.py').read_text()
            rendered = template.replace('__TW_VOICE_BATCH_CONFIG__',base64.b64encode(json.dumps(batch_conf).encode()).decode())
            rendered = rendered.replace('__TW_VOICE_BATCH_SOURCE__',base64.b64encode(build_source(ROOT)).decode())
            (kernel_dir/'kernel_main.py').write_text(rendered)
        metadata_path = kernel_dir / 'kernel-metadata.json'
        metadata = json.loads(metadata_path.read_text())
        # Kaggle can derive a new kernel's URL suffix from its title.
        metadata['title'] = slug.replace('-', ' ')
        metadata_path.write_text(json.dumps(metadata, indent=2))
        summary['kernel'] = prepared['kernel']
        args = ['kaggle', 'kernels', 'push', '-p', str(kernel_dir), '--timeout',
                '9600' if phase=='voice_batch' else '3600']
        if phase in {'image', 'voice', 'voice_batch'}:
            args += ['--accelerator', 'NvidiaTeslaT4']
        pushed = command(args)
        (result_dir / 'submission_response.txt').write_text(pushed)
        import re
        urls = re.findall(r'https://www\.kaggle\.com/code/(wolf19loner92/[a-z0-9-]+)', pushed)
        if urls:
            prepared['kernel'] = urls[-1]
            summary['kernel'] = prepared['kernel']
        summary['submitted'] = True
        try:
            summary['kernel_status'] = poll_kernel(prepared['kernel'],
                timeout_seconds=5400 if phase=='voice_batch' else 3500, interval_seconds=45)
        except Exception as exc:
            summary['kernel_error'] = type(exc).__name__ + ': ' + str(exc)
        command(['kaggle', 'kernels', 'output', prepared['kernel'], '-p', str(result_dir),
                 '--force', '--file-pattern', r'(TwoWorlds_PRIVATE_.*\.zip|two_worlds_kaggle_receipt\.json|batch_voice_.*\.json)$'],
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
        seal(archive, ROOT / '.github/batch03_requests/recipient_public.pem',
             ROOT / 'dist/private_batch03/production_result.twenc',
             {'purpose': 'batch03-production', 'nonce': conf['nonce']})
        print(json.dumps({k: summary[k] for k in ['nonce', 'phase', 'story', 'paid_services', 'published']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'produce'])
    getattr(__import__(__name__, fromlist=['']), parser.parse_args().action)()
