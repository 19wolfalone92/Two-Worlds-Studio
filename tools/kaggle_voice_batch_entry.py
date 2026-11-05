"""Private six-story voice pass: two independent 16GB T4 workers, no pooled VRAM."""
from __future__ import annotations
import base64
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

CONFIG_B64 = '__TW_VOICE_BATCH_CONFIG__'
SOURCE_B64 = '__TW_VOICE_BATCH_SOURCE__'
OUT = Path('/kaggle/working')
BASE = OUT/'tw_voice_modules'


def resolve_source(hint: Path, dataset_slug: str, input_root: Path = Path('/kaggle/input')) -> Path:
    if hint.is_file():
        return hint
    found = [p for p in input_root.rglob(hint.name) if dataset_slug in p.parts and p.is_file()]
    if len(found)==1:
        return found[0]
    projects = [p for p in input_root.rglob('project.json')
                if p.parent.name=='13' and p.parent.parent.name=='production_queue'
                and dataset_slug in p.parts]
    if len(projects)==1:
        root = projects[0].parent.parent.parent
        if all((root/'production_queue'/str(i)/'project.json').is_file() for i in range(13,19)):
            return root
    raise FileNotFoundError('Locked private six-story source is missing or ambiguous')


def find_recovery(input_root: Path, story: int, model_kernel_slug: str) -> list[Path]:
    """Model caches may contain old story media; never import them as recovery."""
    return [p for p in input_root.rglob(f'TwoWorlds_PRIVATE_{story}_voice.zip')
            if model_kernel_slug not in p.parts]


def voice_tasks(conf: dict) -> list[dict]:
    tasks = conf.get('tasks', [{'story': n, 'first_scene': 1, 'count': 8}
                               for n in range(13, 19)])
    if (not isinstance(tasks, list) or not 1 <= len(tasks) <= 6
            or any(not isinstance(t, dict) or type(t.get('story')) is not int
                   or t['story'] not in range(13, 19)
                   or type(t.get('first_scene')) is not int
                   or type(t.get('count')) is not int
                   or not 1 <= t['first_scene'] <= 8
                   or not 1 <= t['count'] <= 8
                   or t['first_scene'] + t['count'] > 9 for t in tasks)
            or len({t['story'] for t in tasks}) != len(tasks)):
        raise ValueError('Invalid bounded per-story voice tasks')
    return tasks


def main():
    conf = json.loads(base64.b64decode(CONFIG_B64))
    tasks = voice_tasks(conf)
    BASE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as z:
        for item in z.infolist():
            p = Path(item.filename)
            if p.is_absolute() or '..' in p.parts or p.parts[0] != 'studio':
                raise ValueError('Unexpected source bundle path')
        z.extractall(BASE)
    sys.path.insert(0, str(BASE))
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'qwen-tts', 'soundfile'],
                   check=True, timeout=1200)
    import torch
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError('Real voice batch requires CUDA, not a simulated backend')
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    source = resolve_source(Path(conf['source_path']), conf['dataset_slug'])
    model_root = Path(conf['model_root'])
    if not (model_root/'voice/config.json').is_file():
        found = [p.parent.parent for p in Path('/kaggle/input').rglob('config.json')
                 if p.parent.name == 'voice' and conf['model_kernel_slug'] in p.parts]
        if len(found) != 1:
            raise FileNotFoundError('Previously validated Qwen model cache is missing')
        model_root = found[0]
    # Each process loads one model on its own card. No device has 32GB VRAM.
    assignments = [tasks[i::min(gpu_count,2)] for i in range(min(gpu_count,2))]
    state = {'schema':1, 'status':'RUNNING', 'gpu_names':gpu_names, 'memory_pooled':False,
             'stories':{}, 'paid_services':False, 'published':False}
    started = time.monotonic()
    processes = {}
    logs = {}
    def checkpoint():
        state['elapsed_s'] = round(time.monotonic()-started,2)
        (OUT/'batch_voice_status.json').write_text(json.dumps(state,indent=2))
    def launch(gpu):
        if not assignments[gpu]:
            return
        task = assignments[gpu].pop(0)
        story = task['story']
        out = OUT/f'story_{story}'
        out.mkdir(exist_ok=True)
        recovery = find_recovery(Path('/kaggle/input'),story,conf['model_kernel_slug'])
        args = [sys.executable,'-m','studio.kaggle_shorts_worker',
                '--source',str(source),'--story',str(story),'--phase','voice',
                '--models',str(model_root),'--work',str(OUT/'private_projects'/str(story)),
                '--out',str(out),'--start',str(task['first_scene']),
                '--count',str(task['count']),'--gpu-id',str(gpu),
                '--time-budget','4800','--speaker',conf.get('speakers',{}).get(str(story),'Ryan' if story == 16 else 'Serena')]
        if len(recovery)==1:
            args += ['--recovery',str(recovery[0])]
        env = os.environ.copy()
        env['PYTHONPATH'] = str(BASE)
        log = (out/'private_worker.log').open('w')
        proc = subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)
        processes[gpu] = (proc,story)
        logs[gpu] = log
        state['stories'][str(story)] = {'status':'RUNNING','gpu':gpu}
        checkpoint()
    for gpu in range(len(assignments)):
        launch(gpu)
    while processes:
        for gpu,(proc,story) in list(processes.items()):
            code = proc.poll()
            if code is None:
                continue
            logs.pop(gpu).close()
            del processes[gpu]
            receipt = OUT/f'story_{story}'/'two_worlds_kaggle_receipt.json'
            success = False
            if receipt.is_file():
                item = json.loads(receipt.read_text())
                success = bool(item.get('success') and item.get('phase_complete'))
            state['stories'][str(story)] = {'status':'COMPLETE' if success else 'PARTIAL_OR_FAILED',
                                           'returncode':code,'gpu':gpu}
            checkpoint()
            launch(gpu)
        if time.monotonic()-started > 9000:
            for proc,story in processes.values():
                proc.terminate()
            raise TimeoutError('Batch reached its explicit 150-minute compute deadline')
        time.sleep(5)
    state['status'] = ('COMPLETE' if all(v['status']=='COMPLETE' for v in state['stories'].values())
                       and len(state['stories'])==len(tasks) else 'PARTIAL')
    checkpoint()


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        (OUT/'batch_voice_failure.json').write_text(json.dumps({'success':False,
            'type':type(exc).__name__,'message':str(exc),'published':False}))
        raise
