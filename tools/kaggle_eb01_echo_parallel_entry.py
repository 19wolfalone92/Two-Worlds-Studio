"""One private Kaggle notebook, shared T5 precompute, two independent GPU workers."""
import base64,json,os,re,subprocess,sys,time,traceback
from pathlib import Path
CONFIG_B64='__EB01_ECHO_PARALLEL_CONFIG_B64__'


def main():
    began=time.monotonic();os.umask(0o077);out=Path('/kaggle/working');out.mkdir(exist_ok=True)
    cfg=json.loads(base64.b64decode(CONFIG_B64,validate=True))
    if (cfg.get('private') is not True or cfg.get('no_paid_services') is not True or
        cfg.get('production_jobs')!=0 or cfg.get('benchmark_only') is not True or
        cfg.get('source_dataset')!='wolf19loner92/eb01-v7-private-production-sources' or cfg.get('account_alias') not in ('owner','collaborator') or
        not re.fullmatch('[a-f0-9]{40}',cfg.get('revision','')) or
        cfg.get('branch')!='feature/eb01-audio-driven-actors-20261120'):
        raise ValueError('Invalid bounded private benchmark')
    root=Path('/kaggle/temp/eb01-echo-parallel');root.mkdir(parents=True,exist_ok=True)
    log=out/'EB01_ECHO_PRIVATE_SETUP.log';studio=root/'studio';runtime=root/'runtime'
    def command(argv,timeout):
        with log.open('ab') as f:subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=f,
            stderr=subprocess.STDOUT,check=True,timeout=timeout)
    try:
        print('EB01_PARALLEL_PRIVATE_BOOTSTRAP_BEGIN',flush=True)
        command(['git','clone','--depth','1','--branch',cfg['branch'],
                 'https://github.com/19wolfalone92/Two-Worlds-Studio.git',str(studio)],180)
        command(['git','-C',str(studio),'fetch','--depth','1','origin',cfg['revision']],120)
        command(['git','-C',str(studio),'checkout','--detach',cfg['revision']],40)
        sys.path.insert(0,str(studio))
        from studio.eb01_echo_profile import UPSTREAM,UPSTREAM_SHA
        from studio.eb01_anatomy_profile import select_specs
        select_specs(cfg)
        if cfg['upstream_revision']!=UPSTREAM_SHA:raise ValueError('Wrong pinned model profile')
        command(['git','clone','--depth','1',UPSTREAM,str(runtime)],180)
        command(['git','-C',str(runtime),'fetch','--depth','1','origin',UPSTREAM_SHA],120)
        command(['git','-C',str(runtime),'checkout','--detach',UPSTREAM_SHA],40)
        command([sys.executable,'-m','pip','install','--quiet','--disable-pip-version-check',
                 '--only-binary=numpy,sentencepiece,scipy,tokenizers,safetensors,numba,llvmlite,opencv-python-headless',
                 '-r',str(studio/'requirements-echo-runtime.txt')],420)
        cfg.update(work=str(root),runtime=str(runtime),studio_revision=cfg['revision'])
        config=root/'private-benchmark-config.json';config.write_text(json.dumps(cfg))
        os.environ.update(PYTHONPATH=str(studio),PYTHONUNBUFFERED='1',USE_TF='0',USE_FLAX='0',
            HF_HOME=str(root/'hf-cache'),HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1',
            TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        os.chdir(studio)
        command([sys.executable,'-u','-m','tools.eb01_echo_parallel_prepare','--config',str(config)],900)
        print('EB01_TEXT_PROCESS_EXITED_START_INDEPENDENT_GPU_WORKERS',flush=True)
        from tools.eb01_bounded_inference import run_native_bounded
        run_native_bounded([sys.executable,'-u','-m','tools.eb01_echo_parallel_batch','--config',str(config)],
            cwd=studio,env=dict(os.environ),log_path=out/'EB01_ECHO_PRIVATE_PARALLEL.log',
            deadline_seconds=min(2650,max(10,3400-(time.monotonic()-began))),heartbeat_seconds=20)
        print('EB01_PARALLEL_BENCHMARK_FINISHED',flush=True)
    except BaseException as exc:
        with log.open('a') as f:traceback.print_exc(file=f)
        (out/'EB01_ECHO_PARALLEL_FAILURE_SAFE.json').write_text(json.dumps({'schema':1,
            'status':'PRIVATE_PARALLEL_BENCHMARK_FAILED','error_type':type(exc).__name__,
            'whole_movie_approved':False,'success':False}))
        print('EB01_PARALLEL_PRIVATE_FAILURE',type(exc).__name__,flush=True)
        raise SystemExit(1) from None

if __name__=='__main__':main()
