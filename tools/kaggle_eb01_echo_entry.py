"""Private Kaggle bootstrap; all raw diagnostics remain private kernel output."""
import base64,json,os,re,subprocess,sys,traceback
from pathlib import Path

CONFIG_B64='__EB01_ECHO_CONFIG_B64__'


def main():
    os.umask(0o077)
    out=Path('/kaggle/working');out.mkdir(exist_ok=True)
    log=out/'EB01_ECHO_PRIVATE_SETUP.log'
    cfg=json.loads(base64.b64decode(CONFIG_B64,validate=True))
    if (cfg.get('private') is not True or cfg.get('no_paid_services') is not True or
        cfg.get('allow_social_publish') is not False or cfg.get('new_gpu_jobs')!=1 or
        cfg.get('branch')!='feature/eb01-audio-driven-actors-20261120' or
        cfg.get('source_dataset')!='wolf19loner92/eb01-v7-private-production-sources' or
        cfg.get('account_alias') not in ('owner','collaborator') or
        not re.fullmatch('[0-9a-f]{40}',cfg.get('revision',''))):
        raise ValueError('Invalid bounded private audio pilot')
    root=Path('/kaggle/temp/eb01-echo');root.mkdir(parents=True,exist_ok=True)
    studio=root/'studio';runtime=root/'runtime'
    def command(argv,timeout):
        with log.open('ab') as f:
            subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True,timeout=timeout)
    try:
        print('EB01_ECHO_PRIVATE_BOOTSTRAP_BEGIN',flush=True)
        command(['git','clone','--depth','1','--branch',cfg['branch'],
                 'https://github.com/19wolfalone92/Two-Worlds-Studio.git',str(studio)],180)
        command(['git','-C',str(studio),'fetch','--depth','1','origin',cfg['revision']],120)
        command(['git','-C',str(studio),'checkout','--detach',cfg['revision']],40)
        if subprocess.check_output(['git','-C',str(studio),'rev-parse','HEAD'],text=True).strip()!=cfg['revision']:
            raise ValueError('Studio revision mismatch')
        sys.path.insert(0,str(studio))
        from studio.eb01_echo_profile import UPSTREAM,UPSTREAM_SHA,PROFILE
        if cfg['upstream_revision']!=UPSTREAM_SHA or cfg['profile']!=PROFILE:
            raise ValueError('Unapproved engine revision or profile')
        command(['git','clone','--depth','1',UPSTREAM,str(runtime)],180)
        command(['git','-C',str(runtime),'fetch','--depth','1','origin',UPSTREAM_SHA],120)
        command(['git','-C',str(runtime),'checkout','--detach',UPSTREAM_SHA],40)
        # Keep preinstalled GPU PyTorch/CUDA. The dependency set excludes paid
        # services and unused face detectors, and does not install CUDA wheels.
        command([sys.executable,'-m','pip','install','--quiet','--disable-pip-version-check',
                 '-r',str(studio/'requirements-echo-runtime.txt')],420)
        from tools.eb01_bounded_inference import run_native_bounded
        cfg.update(runtime=str(runtime),work=str(root),studio_revision=cfg['revision'])
        config=root/'private-native-config.json';config.write_text(json.dumps(cfg))
        env=dict(os.environ)
        env.update(PYTHONPATH=str(studio),PYTHONUNBUFFERED='1',HF_HOME=str(root/'hf-cache'),
                   HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1',
                   TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        run_native_bounded([sys.executable,'-u','-m','tools.eb01_echo_native_infer','--config',str(config)],
            cwd=studio,env=env,log_path=out/'EB01_ECHO_PRIVATE_NATIVE.log',
            deadline_seconds=3000,heartbeat_seconds=20)
        print('EB01_ECHO_NATIVE_PILOT_FINISHED',flush=True)
    except BaseException as exc:
        with log.open('a') as f:traceback.print_exc(file=f)
        from_stage='NO_NATIVE_STAGE_MARKER'
        try:
            from tools.eb01_bounded_inference import latest_stage
            from_stage=latest_stage(out/'EB01_ECHO_PRIVATE_NATIVE.log')
        except ImportError:pass
        (out/'EB01_ECHO_FAILURE_SAFE.json').write_text(json.dumps({
            'schema':1,'status':'PRIVATE_NATIVE_PILOT_FAILED','error_type':type(exc).__name__,
            'last_stage':from_stage,'success':False,'whole_movie_approved':False}))
        print('EB01_ECHO_PRIVATE_FAILURE',type(exc).__name__,from_stage,flush=True)
        # No traceback reaches notebook stdout or a public Actions log.

if __name__=='__main__':main()
