"""Read official accelerator quota and bounded live stage labels; never submit a job."""
from __future__ import annotations
import argparse,ast,base64,hashlib,json,math,os,re,subprocess,tempfile
from pathlib import Path
from tools.eb01_native_two_account_pilot import call,PilotBlocked,STATE
from tools.eb01_kaggle2_auth_probe import query,SOURCE_DATASET
from tools.eb01_bounded_inference import latest_stage
from tools.eb01_kaggle_token_env import token_environment


def parse_quota(raw):
    rows=json.loads(raw)
    if not isinstance(rows,list) or len(rows)>4:raise ValueError('Invalid quota table')
    result={}
    for item in rows:
        resource=item.get('resource')
        if resource not in ('GPU','TPU') or resource in result:raise ValueError('Unexpected resource')
        values={}
        for key in ('used','remaining','total'):
            value=item[key]
            if not isinstance(value,str) or not re.fullmatch(r'\d+(?:\.\d+)?h',value):raise ValueError('Nonfinite quota')
            n=float(value[:-1])
            if not math.isfinite(n) or n>10000:raise ValueError('Invalid quota')
            values[key+'_hours']=n
        refresh=item.get('refreshAt','')
        if not isinstance(refresh,str) or len(refresh)>40 or refresh and not re.fullmatch(r'[0-9T:+.Z-]+',refresh):
            raise ValueError('Unexpected quota refresh timestamp')
        values['refresh_at']=refresh;result[resource]=values
    if 'GPU' not in result:raise ValueError('No actual GPU quota returned')
    return result


def observe(alias,token,source_run):
    if alias not in ('owner','collaborator') or not re.fullmatch(r'[1-9][0-9]{5,20}',source_run):
        raise ValueError('Invalid immutable source')
    report={'schema':1,'account_alias':alias,'source_run':source_run,'gpu_submissions':0,
            'quota_status':'UNVERIFIED','quality_approved':False,'whole_movie_approved':False}
    auth,handle=query(token)
    report['authentication']=auth
    if auth!='AUTHENTICATED_READ_ONLY' or not handle:return report
    try:
        report['accelerator_quota']=parse_quota(call(token,['kaggle','quota','--format','json'],timeout=65))
        report['quota_status']='ACTUAL_OFFICIAL_API_QUOTA_READ'
    except (PilotBlocked,ValueError,KeyError,TypeError):report['quota_status']='OFFICIAL_QUOTA_COMMAND_OR_ACCESS_UNAVAILABLE'
    slug='eb01-echo-'+hashlib.sha256(('eb01-echo-v1-'+source_run+'-'+alias).encode()).hexdigest()[:16]
    kernel=handle+'/'+slug
    try:
        with tempfile.TemporaryDirectory(dir=os.environ['RUNNER_TEMP'],prefix='eb01-server-readback-private-') as temp:
            folder=Path(temp)
            call(token,['kaggle','kernels','pull',kernel,'-p',str(folder),'-m'],timeout=65)
            meta=json.loads((folder/'kernel-metadata.json').read_text())
            code=Path(meta['code_file'])
            if code.name!=str(code):raise ValueError('Unexpected server source path')
            tree=ast.parse((folder/code).read_text())
            configs=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='CONFIG_B64' for t in n.targets)]
            if len(configs)!=1:raise ValueError('Server source config missing or ambiguous')
            cfg=json.loads(base64.b64decode(configs[0],validate=True))
            source=json.loads(Path('ops/eb01_echo_collect.json').read_text())
            report['server_private_kernel_verified']=meta.get('is_private') is True
            report['server_shared_dataset_attached']=meta.get('dataset_sources')==[SOURCE_DATASET]
            report['server_pinned_source_verified']=(cfg.get('revision')==source['source_revision'] and
                cfg.get('account_alias')==alias and cfg.get('private') is True and cfg.get('no_paid_services') is True)
            report['server_requested_t4_shape_verified']=meta.get('machine_shape')=='NvidiaTeslaT4'
    except (PilotBlocked,ValueError,KeyError,TypeError,OSError):report['server_readback_status']='UNVERIFIED'
    try:
        states=STATE.findall(call(token,['kaggle','kernels','status',kernel],timeout=65))
        report['kaggle_state']=states[0] if len(states)==1 else 'UNRECOGNIZED_STATUS'
    except PilotBlocked:report['kaggle_state']='STATUS_READ_UNAVAILABLE'
    try:
        raw=call(token,['kaggle','kernels','logs',kernel],timeout=65)
        if report.get('kaggle_state')=='RUNNING':
            # The persisted blob may be empty until completion. The official
            # live SSE stream replays current log lines; stop after 18 seconds.
            with token_environment(token) as env:
                try:
                    p=subprocess.run(['kaggle','kernels','logs',kernel,'--follow'],env=env,
                        capture_output=True,stdin=subprocess.DEVNULL,timeout=18)
                    stream=p.stdout
                except subprocess.TimeoutExpired as exc:stream=exc.stdout or b''
            raw+='\n'+stream.decode('utf-8',errors='replace')
            report['bounded_live_stream_observed']=True
        report['bootstrap_started']='EB01_ECHO_PRIVATE_BOOTSTRAP_BEGIN' in raw
        # Never echo notebook logs: only stage allowlist and numeric times leave temp.
        with tempfile.TemporaryDirectory(dir=os.environ['RUNNER_TEMP'],prefix='eb01-stage-private-') as temp:
            log=Path(temp)/'notebook.log';log.write_text(raw)
            report['last_native_stage']=latest_stage(log)
        markers=re.findall(r'EB01_GPU_STAGE_HEARTBEAT\s+stage=([A-Z0-9_]+)\s+seconds=([0-9.]+)',raw)
        if markers:
            report['last_bounded_worker_seconds']=float(markers[-1][1])
            report['last_native_stage']=markers[-1][0] if markers[-1][0] in {
                'NO_NATIVE_STAGE_MARKER','BEGIN_MODEL_LOAD','END_MODEL_LOAD','BEGIN_CONDITIONING_VAE',
                'BEGIN_TRANSFORMER','DENOISE_STEP_START','BEGIN_DECODE','NATIVE_VIDEO_OUTPUT_READY'} else 'UNKNOWN_STAGE'
        report['private_failure_marker_seen']='EB01_ECHO_PRIVATE_FAILURE' in raw
    except PilotBlocked:report['live_stage_status']='NOTEBOOK_LOGS_API_UNAVAILABLE'
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--account',choices=('owner','collaborator'),required=True)
    args=p.parse_args();os.umask(0o077)
    source=json.loads(Path('ops/eb01_echo_collect.json').read_text())['source_run']
    r=observe(args.account,os.environ.get('EB01_ACCOUNT_TOKEN',''),source)
    dest=Path(os.environ['RUNNER_TEMP'])/('EB01_QUOTA_'+args.account+'_SAFE.json')
    dest.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print('EB01_OFFICIAL_QUOTA_AND_STAGE_NUMERIC',json.dumps(r,sort_keys=True))

if __name__=='__main__':main()
