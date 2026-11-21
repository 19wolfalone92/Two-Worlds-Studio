"""One bounded private VAE-only restore; retries collect the same immutable job."""
from __future__ import annotations
import base64,hashlib,json,os,re,subprocess,tempfile,time,zipfile
from pathlib import Path
from studio.eb01_echo_profile import BRANCH,UPSTREAM_SHA,PROFILE,pilot_spec
from studio.sealed_artifacts import seal
from tools.eb01_echo_restore_checkpoint import SOURCE_RUN,LATENT_SHA,ORIGINAL_MP4_SHA,VAE_SHA
from tools.eb01_native_two_account_pilot import call,PilotBlocked,STATE
from tools.eb01_kaggle2_auth_probe import query,SOURCE_DATASET
from tools.eb01_kaggle_quota_observer import parse_quota


def prepare(root,handle,run_id,revision):
    if (not re.fullmatch(r'[1-9][0-9]{5,20}',run_id) or not re.fullmatch('[a-f0-9]{40}',revision) or
        not re.fullmatch(r'[A-Za-z0-9_-]{2,30}',handle)):raise PilotBlocked('INVALID_IMMUTABLE_RESTORE_IDENTITY')
    source_slug='eb01-echo-'+hashlib.sha256(('eb01-echo-v1-'+SOURCE_RUN+'-collaborator').encode()).hexdigest()[:16]
    source_kernel=handle+'/'+source_slug
    slug='eb01-restore-'+hashlib.sha256(('eb01-restore-v1-'+run_id).encode()).hexdigest()[:16]
    cfg={'private':True,'no_paid_services':True,'allow_social_publish':False,'new_gpu_jobs':1,
        'restoration_only':True,'source_run':SOURCE_RUN,'branch':BRANCH,'revision':revision,
        'source_dataset':SOURCE_DATASET,'profile':PROFILE,'account_alias':'collaborator','upstream_revision':UPSTREAM_SHA}
    text=Path('tools/kaggle_eb01_echo_entry.py').read_text()
    for old,new in [('__EB01_ECHO_CONFIG_B64__',base64.b64encode(json.dumps(cfg,sort_keys=True).encode()).decode()),
        ('/kaggle/temp/eb01-echo','/kaggle/temp/eb01-echo-restore'),
        ("'tools.eb01_echo_native_infer'","'tools.eb01_echo_restore_checkpoint'"),
        ('deadline_seconds=3000','deadline_seconds=900')]:
        if text.count(old)!=1:raise PilotBlocked('PINNED_RESTORE_BOOTSTRAP_BOUNDARY_CHANGED')
        text=text.replace(old,new)
    folder=root/'kernel';folder.mkdir(mode=0o700);(folder/'kernel_main.py').write_text(text)
    kernel=handle+'/'+slug
    (folder/'kernel-metadata.json').write_text(json.dumps({'id':kernel,'title':slug,'code_file':'kernel_main.py',
        'language':'python','kernel_type':'script','is_private':'true','enable_gpu':'true','enable_internet':'true',
        'machine_shape':'NvidiaTeslaT4','dataset_sources':[SOURCE_DATASET],
        'competition_sources':[],'kernel_sources':[source_kernel],'model_sources':[]}))
    return folder,kernel,source_kernel


def verify(output,revision):
    import numpy as np
    from studio.comfy_neural import _probe
    from tools.eb01_native_pixel_evidence import decoded_pixel_evidence
    from studio.eb01_motion_artifact_gate import inspect_clip
    restored=output/'EB01_RESTORED_RUSSIAN_NATIVE.mp4';source=output/'EB01_RESTORE_SOURCE_COMPARE_ONLY.mp4'
    if not restored.is_file():return {'technical_pass':False,'actual_restored_mp4':False}
    proof=json.loads((output/'EB01_RESTORE_RECEIPT_SAFE.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    meta=_probe(restored);pixels=decoded_pixel_evidence(restored,frames=65)
    # Independent original RGB comparison; neither input is modified or repaired.
    def raw(p):return subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-an','-pix_fmt','rgb24','-f','rawvideo','-'],timeout=120)
    a=np.frombuffer(raw(source),np.uint8);b=np.frombuffer(raw(restored),np.uint8)
    same_shape=a.shape==b.shape
    mae=round(float(np.abs(a.astype(np.int16)-b.astype(np.int16)).mean()),6) if same_shape else None
    audio=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','a:0','-show_entries',
        'stream=duration','-of','json',str(restored)],timeout=45))['streams']
    checks={'source_original_sha':sha(source)==ORIGINAL_MP4_SHA,'restored_sha':sha(restored)==proof.get('restored_mp4_sha256'),
        'saved_latent_sha':proof.get('checkpoint_sha256')==LATENT_SHA,'original_vae_sha':proof.get('vae_weights_sha256')==VAE_SHA,
        'pinned_restore_revision':proof.get('studio_revision')==revision,
        'native_geometry_and_clock':meta['width']==576 and meta['height']==1024 and meta['fps']=='25' and meta['frames']==65,
        'unique_original_frames':pixels['unique_decoded_rgb_frames']==65,
        'original_full_voice':proof.get('original_voice_sha256')==pilot_spec('collaborator')['voice_sha256'] and
            len(audio)==1 and float(audio[0]['duration'])>=2.16,
        'no_new_diffusion_or_models':proof.get('new_denoising_steps')==0 and proof.get('diffusion_model_loaded') is False and
            proof.get('text_encoder_loaded') is False and proof.get('acoustic_encoder_loaded') is False,
        'original_rgb_restored_within_half_level':same_shape and mae is not None and mae<=.5,
        'no_neon':inspect_clip(restored,expected_fps=25)['status']=='NO_NEON_ARTIFACT_FOUND'}
    return {'technical_pass':all(checks.values()),'actual_restored_mp4':True,'checks':checks,**pixels,
        'probe':meta,'decoded_rgb_mean_absolute_error':mae,'decoded_rgb_identical':same_shape and np.array_equal(a,b),
        'mp4_bytes_identical':sha(source)==sha(restored),'vae_decode_seconds':proof.get('vae_decode_seconds'),
        'resources':proof.get('resources'),'restoration_only':True,'new_original_takes':0,'quality_approved':False}


def run(token,root,run_id,revision,attempt,poll_seconds=1800):
    report={'schema':1,'run_id':run_id,'account_alias':'collaborator','source_pilot_run':SOURCE_RUN,
        'restoration_only':True,'gpu_submission_attempts':0,'production_jobs_submitted':0,'new_original_takes':0,
        'quality_approved':False,'whole_movie_approved':False}
    try:
        auth,handle=query(token)
        if auth!='AUTHENTICATED_READ_ONLY' or not handle:raise PilotBlocked('ACCOUNT_AUTH_NOT_VERIFIED')
        folder,kernel,original=prepare(root,handle,run_id,revision)
        if attempt==1:
            pulled=root/'source-readback';pulled.mkdir(mode=0o700)
            call(token,['kaggle','kernels','pull',original,'-p',str(pulled),'-m'],timeout=80)
            meta=json.loads((pulled/'kernel-metadata.json').read_text())
            if meta.get('is_private') is not True:raise PilotBlocked('SOURCE_NOTEBOOK_PRIVACY_UNVERIFIED')
            quota=parse_quota(call(token,['kaggle','quota','--format','json'],timeout=65))['GPU']
            report['official_quota_before']=quota
            if quota['remaining_hours']<.35:raise PilotBlocked('INSUFFICIENT_FREE_QUOTA_FOR_RESTORE')
            report['gpu_submission_attempts']=1
            call(token,['kaggle','kernels','push','-p',str(folder),'--timeout','1500','--accelerator','NvidiaTeslaT4'],timeout=220)
            pulled_new=root/'restore-readback';pulled_new.mkdir(mode=0o700)
            call(token,['kaggle','kernels','pull',kernel,'-p',str(pulled_new),'-m'],timeout=80)
            meta=json.loads((pulled_new/'kernel-metadata.json').read_text())
            report['server_private_kernel_verified']=meta.get('is_private') is True
            report['server_private_source_kernel_attached']=meta.get('kernel_sources')==[original]
        deadline=time.monotonic()+poll_seconds
        while time.monotonic()<deadline:
            states=STATE.findall(call(token,['kaggle','kernels','status',kernel],timeout=65))
            if len(states)!=1:raise PilotBlocked('UNKNOWN_KERNEL_STATE')
            state=states[0];report['kaggle_state']=state;print('EB01_RESTORE_STATE',state,flush=True)
            if state in ('COMPLETE','ERROR','CANCELLED','CANCEL_ACKNOWLEDGED'):
                output=root/'output';output.mkdir(mode=0o700)
                call(token,['kaggle','kernels','output',kernel,'-p',str(output),'--force'],timeout=300)
                try:report['verification']=verify(output,revision)
                except Exception as exc:report['verification']={'technical_pass':False,'error_type':type(exc).__name__}
                archive=root/'private-return.zip'
                with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
                    for p in output.rglob('*'):
                        if p.is_file() and p.suffix in ('.mp4','.json','.log'):z.write(p,str(p.relative_to(output)))
                seal(archive,Path('ops/eb01_native_pilot_return_public.pem'),Path(os.environ['RUNNER_TEMP'])/'EB01_RESTORE.twenc',
                    {'purpose':'eb01-echo-checkpoint-restore','run_id':run_id,'account':'collaborator'})
                report['encrypted_private_return_created']=True
                report['status']='REAL_FULL_CHECKPOINT_RESTORE_TECHNICAL_PASS' if report['verification']['technical_pass'] else 'PRIVATE_RESTORE_FAILED_OR_QA_BLOCKED'
                return report
            time.sleep(30)
        report['status']='SAME_RESTORE_STILL_RUNNING_COLLECT_ONLY'
    except (PilotBlocked,ValueError,OSError,KeyError,subprocess.SubprocessError) as exc:
        report['status']='BLOCKED_'+(str(exc) if isinstance(exc,PilotBlocked) else type(exc).__name__)
    return report


def main():
    os.umask(0o077)
    with tempfile.TemporaryDirectory(dir=os.environ['RUNNER_TEMP'],prefix='eb01-restore-private-') as temp:
        report=run(os.environ.get('EB01_ACCOUNT_TOKEN',''),Path(temp),os.environ['GITHUB_RUN_ID'],
            os.environ['GITHUB_SHA'],int(os.environ.get('GITHUB_RUN_ATTEMPT','1')))
    (Path(os.environ['RUNNER_TEMP'])/'EB01_RESTORE_SAFE.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('EB01_RESTORE_NUMERICAL_RESULT',json.dumps(report,sort_keys=True))
    return 0 if report['status']=='REAL_FULL_CHECKPOINT_RESTORE_TECHNICAL_PASS' else 2

if __name__=='__main__':raise SystemExit(main())
