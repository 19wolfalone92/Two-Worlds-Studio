"""Create the reviewable native25 proposal and actual disjoint durable queues.

Zero Kaggle calls, zero GPU work, no production authorization. Parallel
capacity is admitted only from independently collected real benchmark data.
"""
from __future__ import annotations
import argparse,hashlib,json,math,re
from pathlib import Path
from studio.eb01_echo_production_manifest import plan
from studio.eb01_echo_parallel_profile import PROFILE
from studio.eb01_echo_ledger import EchoLedger
from studio.eb01_v8_private_mux import safe_private_path


def capacity(report,alias):
    if (report.get('account_alias')!=alias or report.get('profile')!=PROFILE or
        report.get('status')!='TWO_SINGLE_CARD_GENERATORS_AND_CONTINUATIONS_TECHNICAL_PASS'):
        raise ValueError('Actual separate-account native continuation benchmark missing')
    workers=report.get('workers',[]);host=report.get('host_and_both_device_measurement',{})
    if len(workers)!=2 or any(w.get('technical_pass') is not True for w in workers):raise ValueError('Both real card workers required')
    busy=host.get('both_gpus_compute_busy_samples')
    if type(busy) is not int or busy<1:raise ValueError('Simultaneous real compute not measured')
    devices=host.get('per_device',[])
    if (len(devices)!=2 or any(type(d.get('physical_gpu_index')) is not int for d in devices) or
        {d.get('physical_gpu_index') for d in devices}!={0,1} or
        any(type(w.get('physical_gpu_index')) is not int for w in workers) or
        {w.get('physical_gpu_index') for w in workers}!={0,1}):raise ValueError('Two separate physical cards required')
    budgets=[]
    for d in devices:
        if any(type(d.get(k)) not in (int,float) or not math.isfinite(d[k]) or d[k]<=0 for k in ('physical_gib','memory_used_peak_gib')):raise ValueError('Nonfinite per-card measurement')
        worker=next(w for w in workers if w['physical_gpu_index']==d['physical_gpu_index'])
        resource=worker.get('resources',{});cuda=resource.get('per_device',[])
        if resource.get('finished') is not True or len(cuda)!=1 or cuda[0].get('index')!=0:
            raise ValueError('Finished isolated-card CUDA measurement required')
        cuda=cuda[0]
        if any(type(cuda.get(k)) not in (int,float) or not math.isfinite(cuda[k]) or cuda[k]<=0
               for k in ('physical_gib','device_memory_used_peak_gib')):raise ValueError('Nonfinite usable CUDA memory')
        # NVML advertises the board's 15 GiB; CUDA exposes only 14.5622 GiB
        # on the measured T4s. Budget the smaller actual usable allocation,
        # and the larger observed peak from the independent samplers.
        usable=min(d['physical_gib'],cuda['physical_gib'])
        peak=max(d['memory_used_peak_gib'],cuda['device_memory_used_peak_gib'])
        if ('T4' not in d.get('name','') or 'T4' not in cuda.get('name','') or usable<14 or peak+.75>usable):
            raise ValueError('Actual per-card VRAM lacks measured production reserve')
        budgets.append({'physical_gpu_index':d['physical_gpu_index'],'nvml_board_gib':d['physical_gib'],
            'usable_cuda_gib':usable,'measured_used_peak_gib':peak,'remaining_reserve_gib':round(usable-peak,4)})
    available=host.get('host_available_min_gib',0)
    if type(available) not in (int,float) or not math.isfinite(available) or available<2:raise ValueError('Host RAM lacks measured production reserve')
    rates=[]
    for w in workers:
        seconds=w.get('native_inference_seconds');frames=w.get('probe',{}).get('frames')
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0 or type(frames) is not int or frames<33:
            raise ValueError('Actual native compute measurement missing')
        rates.append(seconds/frames)
    return {'verified_parallel_processes':2,'worst_measured_gpu_seconds_per_frame':max(rates),
        'source_benchmark_run':report.get('run_id'),'source_report_sha256':hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest(),
        'measured_host_and_per_device':host,'actual_usable_per_card_budget':budgets,'human_quality_approved':False}


def prepare(output,owner_report,collaborator_report):
    output=safe_private_path(output)
    if output.exists():raise FileExistsError('Do not overwrite an immutable production preparation')
    caps={a:capacity(r,a) for a,r in (('owner',owner_report),('collaborator',collaborator_report))}
    doc=plan(profile=PROFILE,verified_parallel_processes=2)
    output.mkdir(parents=True,mode=0o700)
    summaries={};estimates={}
    for alias in ('owner','collaborator'):
        q=EchoLedger(output/(alias+'.sqlite'),alias,doc)
        summaries[alias]=q.summary();q.close()
        frames=sum(j['native_frames'] for j in doc['jobs'] if j['account_alias']==alias)
        estimates[alias]={'native_frames_to_generate':frames,'estimated_session_seconds_with_50pct_reserve_and_10min_startup':
            round(caps[alias]['worst_measured_gpu_seconds_per_frame']*frames/2*1.5+600,1),
            'estimate_is_not_a_full_movie_benchmark':True}
    (output/'NATIVE25_30_ORIGINAL_JOBS.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    gate={'schema':1,'status':'PREPARED_PRIVATE_QUEUES_QUALITY_RELEASE_STILL_BLOCKED',
        'queues':summaries,'actual_benchmark_capacity':caps,'resource_projection':estimates,
        'fresh_quota_recheck_at_execution_required':True,'full_original_voice_sha_recheck_at_execution_required':True,
        'production_execution_authorized':False,'quality_confirmation_required':True,'gpu_jobs_submitted':0,
        'main_merge_performed':False,'movie_ready':False,'published':False}
    (output/'PRODUCTION_GATE_SAFE.json').write_text(json.dumps(gate,indent=2,sort_keys=True)+'\n')
    return gate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--owner-numeric',type=Path,required=True);p.add_argument('--collaborator-numeric',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.output,json.loads(a.owner_numeric.read_text()),json.loads(a.collaborator_numeric.read_text())),sort_keys=True))
