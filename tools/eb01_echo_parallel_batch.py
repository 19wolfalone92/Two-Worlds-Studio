"""Bounded two-process/two-card benchmark with real host and NVML telemetry."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path


def batch(config):
    import psutil
    from tools.eb01_bounded_inference import latest_stage
    cfg=json.loads(Path(config).read_text());out=Path('/kaggle/working')
    started=time.monotonic();workers={};logs={};states={};last={};gpu_peaks={}
    telemetry={'schema':1,'samples':0,'host_used_peak_gib':0.,'host_available_min_gib':1000.,
        'worker_rss_sum_peak_gib':0.,'both_gpus_compute_busy_samples':0,'per_device':[],
        'two_separate_cards_not_pooled':True,'t5_process_exited_before_workers':True,'quality_approved':False}
    def launch(slot):
        folder=out/('slot'+str(slot));folder.mkdir(exist_ok=True)
        log=(folder/'EB01_ECHO_PRIVATE_NATIVE.log').open('wb');logs[slot]=log
        env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=str(slot)
        process=subprocess.Popen([sys.executable,'-u','-m','tools.eb01_echo_single_card_worker',
            '--config',str(config),'--slot',str(slot)],env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers[slot]=(process,time.monotonic());states[slot]='RUNNING'
        print('EB01_PARALLEL_WORKER_STARTED',slot,flush=True)
    def sample():
        vm=psutil.virtual_memory();telemetry['samples']+=1
        telemetry['host_used_peak_gib']=max(telemetry['host_used_peak_gib'],round(vm.used/2**30,4))
        telemetry['host_available_min_gib']=min(telemetry['host_available_min_gib'],round(vm.available/2**30,4))
        total=0.
        for process,_ in workers.values():
            try:total+=psutil.Process(process.pid).memory_info().rss/2**30
            except psutil.NoSuchProcess:pass
        telemetry['worker_rss_sum_peak_gib']=max(telemetry['worker_rss_sum_peak_gib'],round(total,4))
        try:
            data=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True,timeout=8)
            busy=0
            for row in data.splitlines():
                index,name,physical,used,util=[v.strip() for v in row.split(',')]
                i=int(index);physical=float(physical)/1024;used=float(used)/1024;util=float(util)
                item=gpu_peaks.setdefault(i,{'physical_gpu_index':i,'name':name,'physical_gib':round(physical,4),
                    'memory_used_peak_gib':0.,'gpu_utilization_percent_peak':0.})
                item['memory_used_peak_gib']=max(item['memory_used_peak_gib'],round(used,4))
                item['gpu_utilization_percent_peak']=max(item['gpu_utilization_percent_peak'],util)
                if util>=20:busy+=1
            if busy==2:telemetry['both_gpus_compute_busy_samples']+=1
        except (subprocess.SubprocessError,ValueError,OSError):telemetry['nvidia_smi_sample_missing']=True
        telemetry['per_device']=list(gpu_peaks.values());telemetry['wall_seconds']=round(time.monotonic()-started,3)
        (out/'EB01_ECHO_PARALLEL_HOST_SAFE.json').write_text(json.dumps(telemetry,sort_keys=True))
    try:
        launch(0)
        # Avoid two simultaneous large CPU checkpoint initializations. Once
        # the first pipeline is ready, its model process has released temporary
        # full-precision loader tensors; then start the second separate card.
        while not (out/'slot0/WORKER_MODEL_READY_SAFE.json').is_file():
            sample();p,t=workers[0]
            if p.poll() is not None:raise RuntimeError('First independent video loader failed')
            if time.monotonic()-t>900:raise TimeoutError('Independent video loader exceeded budget')
            time.sleep(5)
        launch(1)
        while any(p.poll() is None for p,_ in workers.values()):
            sample()
            for slot,(p,t) in workers.items():
                stage=latest_stage(out/('slot'+str(slot))/'EB01_ECHO_PRIVATE_NATIVE.log')
                if last.get(slot)!=stage:
                    print('EB01_PARALLEL_STAGE',slot,stage,round(time.monotonic()-t,1),flush=True);last[slot]=stage
                    if stage not in ('NO_NATIVE_STAGE_MARKER','NATIVE_VIDEO_OUTPUT_READY'):print('EB01_STAGE_'+stage,flush=True)
                if p.poll() is None and time.monotonic()-t>2400:raise TimeoutError('Bounded independent video budget expired')
            if psutil.virtual_memory().available<1.2*2**30:raise RuntimeError('Host RAM safety reserve exhausted')
            time.sleep(10)
        sample()
        for slot,(p,t) in workers.items():
            states[slot]='COMPLETE' if p.returncode==0 else 'FAILED'
            if p.returncode or latest_stage(out/('slot'+str(slot))/'EB01_ECHO_PRIVATE_NATIVE.log')!='NATIVE_VIDEO_OUTPUT_READY':
                raise RuntimeError('One independent native worker failed')
        result={'schema':1,'account_alias':cfg['account_alias'],'status':'TWO_REAL_SINGLE_CARD_WORKERS_FINISHED',
            'studio_revision':cfg['studio_revision'],'workers':states,'telemetry':telemetry,
            'production_jobs_submitted':0,'whole_movie_approved':False,'success':True}
        (out/'EB01_ECHO_PARALLEL_BATCH_RECEIPT.json').write_text(json.dumps(result,indent=2,sort_keys=True))
        print('EB01_STAGE_NATIVE_VIDEO_OUTPUT_READY',flush=True)
    finally:
        for p,_ in workers.values():
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=8)
                except (ProcessLookupError,subprocess.TimeoutExpired):
                    try:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=8)
                    except (ProcessLookupError,subprocess.TimeoutExpired):pass
        for log in logs.values():log.close()
        sample()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);batch(p.parse_args().config)
