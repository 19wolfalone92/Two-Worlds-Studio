"""One genuine audio-driven private GPU pilot per independent account.

A retry only collects the SAME immutable kernel. Publish numerical checks and
an authenticated encrypted return; never publish voices, frames or raw logs.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,os,re,shutil,subprocess,tempfile,time,zipfile
from pathlib import Path
from studio.eb01_echo_profile import BRANCH,PROFILE,UPSTREAM_SHA,pilot_spec
from studio.sealed_artifacts import seal
from tools.eb01_kaggle2_auth_probe import query,SOURCE_DATASET
from tools.eb01_native_two_account_pilot import call,PilotBlocked,STATE


def prepare(root,alias,handle,run_id,revision,lock):
    spec=pilot_spec(alias)
    if (not re.fullmatch(r'[1-9][0-9]{5,20}',run_id) or
        not re.fullmatch(r'[a-f0-9]{40}',revision) or
        not re.fullmatch(r'[a-zA-Z0-9_-]{2,30}',handle)):
        raise PilotBlocked('INVALID_IMMUTABLE_IDENTITY')
    slug='eb01-echo-'+hashlib.sha256(('eb01-echo-v1-'+run_id+'-'+alias).encode()).hexdigest()[:16]
    cfg={'private':True,'no_paid_services':True,'allow_social_publish':False,'new_gpu_jobs':1,
         'branch':BRANCH,'revision':revision,'source_dataset':SOURCE_DATASET,
         'profile':PROFILE,'account_alias':alias,'model_lock':lock,'upstream_revision':UPSTREAM_SHA}
    text=Path('tools/kaggle_eb01_echo_entry.py').read_text();tag='__EB01_ECHO_CONFIG_B64__'
    if text.count(tag)!=1:raise PilotBlocked('AMBIGUOUS_PRIVATE_KERNEL_CONFIG')
    text=text.replace(tag,base64.b64encode(json.dumps(cfg,sort_keys=True).encode()).decode())
    folder=root/'kernel';folder.mkdir(mode=0o700)
    (folder/'kernel_main.py').write_text(text)
    kernel=handle+'/'+slug
    metadata={'id':kernel,'title':slug,'code_file':'kernel_main.py','language':'python',
        'kernel_type':'script','is_private':'true','enable_gpu':'true','enable_internet':'true',
        'machine_shape':'NvidiaTeslaT4','dataset_sources':[SOURCE_DATASET],
        'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata))
    return folder,kernel


def failure_category(output):
    raw='\n'.join(p.read_text(errors='replace')[-120000:] for p in output.glob('EB01_ECHO_PRIVATE*.log'))
    lower=raw.lower()
    return ('CUDA_OUT_OF_MEMORY' if 'out of memory' in lower else
            'PRIVATE_SOURCE_SHA_OR_VOICE_CLOCK' if 'voice clock' in lower or 'portrait missing' in lower or 'wav sha' in lower else
            'CHECKPOINT_ARCHITECTURE_MISMATCH' if 'completely match' in lower or 'tensor shape mismatch' in lower else
            'MODEL_SETUP_OR_FORWARD_FAILED')


def verify_output(output,alias,revision):
    import numpy as np
    from studio.comfy_neural import _probe
    from studio.eb01_motion_artifact_gate import inspect_clip
    from tools.eb01_native_pixel_evidence import decoded_pixel_evidence
    spec=pilot_spec(alias);video=output/'EB01_ECHO_RUSSIAN_NATIVE.mp4'
    result={'actual_mp4_created':video.is_file(),'technical_pass':False,
            'human_artistic_approved':False,'lipsync_quality_approved':False}
    if not video.is_file():
        result['failure_category']=failure_category(output)
        fail=output/'EB01_ECHO_FAILURE_SAFE.json'
        if fail.is_file():
            d=json.loads(fail.read_text())
            result['last_stage']=d.get('last_stage') if d.get('last_stage') in {
                'NO_NATIVE_STAGE_MARKER','BEGIN_MODEL_LOAD','END_MODEL_LOAD','BEGIN_CONDITIONING_VAE',
                'BEGIN_TRANSFORMER','DENOISE_STEP_START','BEGIN_DECODE'} else 'UNKNOWN_PRIVATE_STAGE'
        return result
    receipt=json.loads((output/'EB01_ECHO_ACTUAL_RECEIPT.json').read_text())
    meta=_probe(video);qc=inspect_clip(video)
    pixels=decoded_pixel_evidence(video,frames=spec['frames'])
    digest=hashlib.sha256(video.read_bytes()).hexdigest()
    decoded=subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],capture_output=True,timeout=100)
    streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)],timeout=45))['streams']
    audios=[s for s in streams if s['codec_type']=='audio']
    gray=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-an','-vf','scale=64:64:flags=area',
                         '-pix_fmt','gray','-f','rawvideo','-'],capture_output=True,check=True,timeout=90).stdout
    frames=np.frombuffer(gray,dtype=np.uint8).reshape(-1,64,64).astype('int16')
    diffs=np.abs(np.diff(frames,axis=0)).mean(axis=(1,2))
    checks={'native_geometry':meta['width']==576 and meta['height']==1024,
        'native_frames':meta['frames']==spec['frames'] and qc['decoded_frames']==spec['frames'],
        'native_25fps':meta['fps']=='25','full_decode':decoded.returncode==0,
        'no_neon':qc['status']=='NO_NEON_ARTIFACT_FOUND',
        'all_original_rgb_frames_distinct':pixels['unique_decoded_rgb_frames']==spec['frames'] and pixels['adjacent_exact_duplicate_pairs']==0,
        'receipt_sha':receipt.get('mp4_sha256')==digest,
        'receipt_source_revision':receipt.get('studio_revision')==revision,
        'receipt_profile':receipt.get('profile')==PROFILE,
        'correct_actor_and_full_voice':receipt.get('shot')==spec['shot'] and
            receipt.get('original_voice_sha256')==spec['voice_sha256'] and receipt.get('original_voice_truncated') is False,
        'audio_actually_conditions_model':receipt.get('audio_is_model_input') is True and
            receipt.get('audio_conditioning_embeddings_shape')==[1,spec['frames'],5,12,768],
        'native_tensor':receipt.get('actual_native_tensor_shape')==[1,3,spec['frames'],1024,576],
        'no_upscale_or_synthetic_frames':receipt.get('source_video_upscaled') is False and receipt.get('synthetic_frames')==0,
        'single_full_length_audio_track':len(audios)==1 and float(audios[0].get('duration',0))>=spec['voice_seconds']}
    result.update(mp4_sha256=digest,bytes=video.stat().st_size,probe=meta,checks=checks,
        technical_pass=all(checks.values()),**pixels,
        motion_proxy={'mean_frame_luma_delta':round(float(diffs.mean()),4),
            'min_frame_luma_delta':round(float(diffs.min()),4),'max_frame_luma_delta':round(float(diffs.max()),4),
            'consecutive_near_duplicate_frame_pairs':int(sum(diffs<.3)),
            'numeric_motion_is_not_proof_of_natural_acting_or_lip_sync':True},
        actual_inference_seconds=receipt.get('native_inference_seconds'),
        process_wall_seconds=receipt.get('process_wall_seconds'))
    resource=output/'EB01_ECHO_RESOURCES_SAFE.json'
    if resource.is_file():
        data=json.loads(resource.read_text())
        result['resources']={k:data.get(k) for k in ('samples','per_device','host_ram_peak_gib',
            'process_rss_peak_gib','cpu_percent_peak','cpu_logical_count','host_ram_total_gib',
            'host_ram_available_gib','disk_free_gib','wall_seconds','finished')}
    return result


def run_pilot(alias,token,root,run_id,revision,attempt,lock,*,collect_run=None,collect_revision=None,poll_seconds=5100):
    report={'schema':1,'account_alias':alias,'profile':PROFILE,'run_id':collect_run or run_id,
            'gpu_submission_attempts':0,'production_jobs_submitted':0,'public_media_uploaded':False,
            'whole_movie_approved':False,'quality_approved':False}
    try:
        auth,handle=query(token)
        if auth!='AUTHENTICATED_READ_ONLY' or not handle:raise PilotBlocked('ACCOUNT_AUTH_NOT_VERIFIED')
        source_run=collect_run or run_id;source_revision=collect_revision or revision
        folder,kernel=prepare(root,alias,handle,source_run,source_revision,lock)
        if attempt==1 and collect_run is None:
            report['gpu_submission_attempts']=1
            call(token,['kaggle','kernels','push','-p',str(folder),'--timeout','3600',
                        '--accelerator','NvidiaTeslaT4'],timeout=220)
        deadline=time.monotonic()+poll_seconds
        while time.monotonic()<deadline:
            found=STATE.findall(call(token,['kaggle','kernels','status',kernel],timeout=65))
            if len(found)!=1:raise PilotBlocked('UNKNOWN_KERNEL_STATE')
            state=found[0];report['kaggle_state']=state
            print('EB01_ECHO_PILOT_STATE',alias,state,flush=True)
            if state in ('COMPLETE','ERROR','CANCELLED','CANCEL_ACKNOWLEDGED'):
                output=root/'output';output.mkdir(mode=0o700)
                call(token,['kaggle','kernels','output',kernel,'-p',str(output),'--force'],timeout=360)
                try:verified=verify_output(output,alias,source_revision)
                except Exception as exc:
                    verified={'actual_mp4_created':(output/'EB01_ECHO_RUSSIAN_NATIVE.mp4').is_file(),
                              'technical_pass':False,'error_type':type(exc).__name__,'private_diagnostics_preserved':True}
                report['native_output']=verified
                # The original frames, complete voice, checkpoint and full diagnostics
                # leave the private runner only inside authenticated encryption.
                selected={'EB01_ECHO_RUSSIAN_NATIVE.mp4','EB01_ECHO_ACTUAL_RECEIPT.json',
                    'EB01_ECHO_DENOISED_LATENTS.safetensors','EB01_ECHO_RESOURCES_SAFE.json',
                    'EB01_ECHO_MODEL_SHA_SAFE.json','EB01_ECHO_FAILURE_SAFE.json',
                    'EB01_ECHO_PRIVATE_NATIVE.log','EB01_ECHO_PRIVATE_SETUP.log'}
                archive=root/'private-return.zip'
                with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
                    for path in output.iterdir():
                        if path.is_file() and path.name in selected:z.write(path,path.name)
                seal(archive,Path('ops/eb01_native_pilot_return_public.pem'),
                    Path(os.environ['RUNNER_TEMP'])/('EB01_ECHO_'+alias+'.twenc'),
                    {'purpose':'eb01-echo-native','run_id':source_run,'account':alias})
                report['encrypted_private_return_created']=True
                report['status']='REAL_AUDIO_DRIVEN_MP4_TECHNICAL_PASS_REVIEW_REQUIRED' if verified['technical_pass'] else 'AUDIO_DRIVEN_PILOT_FAILED_OR_QA_BLOCKED'
                return report
            time.sleep(40)
        report['status']='SAME_PILOT_STILL_QUEUED_OR_RUNNING_COLLECT_ONLY'
    except (PilotBlocked,OSError,ValueError,subprocess.TimeoutExpired) as exc:
        report['status']='BLOCKED_'+(str(exc) if isinstance(exc,PilotBlocked) else type(exc).__name__)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--account',choices=('owner','collaborator'),required=True)
    p.add_argument('--collect-run');p.add_argument('--collect-revision');p.add_argument('--poll-seconds',type=int,default=5100)
    args=p.parse_args();os.umask(0o077)
    if bool(args.collect_run)!=bool(args.collect_revision):raise ValueError('Collect original run AND revision together')
    lock=json.loads(base64.b64decode(os.environ['EB01_MODEL_LOCK_B64'],validate=True))
    with tempfile.TemporaryDirectory(dir=os.environ['RUNNER_TEMP'],prefix='eb01-echo-private-') as temp:
        report=run_pilot(args.account,os.environ.get('EB01_ACCOUNT_TOKEN',''),Path(temp),
            os.environ['GITHUB_RUN_ID'],os.environ['GITHUB_SHA'],int(os.environ.get('GITHUB_RUN_ATTEMPT','1')),lock,
            collect_run=args.collect_run,collect_revision=args.collect_revision,poll_seconds=args.poll_seconds)
    path=Path(os.environ['RUNNER_TEMP'])/('EB01_ECHO_'+args.account+'_SAFE.json')
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('EB01_ECHO_NUMERICAL_RESULT',json.dumps(report,sort_keys=True))
    return 0 if report['status']=='REAL_AUDIO_DRIVEN_MP4_TECHNICAL_PASS_REVIEW_REQUIRED' else 2

if __name__=='__main__':raise SystemExit(main())
