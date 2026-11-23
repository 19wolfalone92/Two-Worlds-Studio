"""Bounded private benchmark: two independent single-card generators per account."""
from __future__ import annotations
import argparse,base64,hashlib,json,os,re,subprocess,tempfile,time,zipfile
from pathlib import Path
from studio.eb01_echo_profile import BRANCH,UPSTREAM_SHA
from studio.eb01_echo_parallel_profile import PROFILE,specs
from studio.sealed_artifacts import seal
from tools.eb01_native_two_account_pilot import call,PilotBlocked,STATE
from tools.eb01_kaggle2_auth_probe import query,SOURCE_DATASET
from tools.eb01_kaggle_quota_observer import parse_quota


def prepare(root,alias,handle,run_id,revision,lock,*,anatomy=False):
    if (alias not in ('owner','collaborator') or not re.fullmatch(r'[1-9][0-9]{5,20}',run_id) or
        not re.fullmatch(r'[a-f0-9]{40}',revision) or not re.fullmatch(r'[A-Za-z0-9_-]{2,30}',handle)):
        raise PilotBlocked('INVALID_IMMUTABLE_BENCHMARK_IDENTITY')
    slug=('eb01-anatomy-' if anatomy else 'eb01-echo-2gpu-')+hashlib.sha256((('eb01-anatomy-v1-' if anatomy else 'eb01-echo-parallel-v1-')+run_id+'-'+alias).encode()).hexdigest()[:14]
    cfg={'private':True,'no_paid_services':True,'production_jobs':0,'benchmark_only':True,
        'source_dataset':SOURCE_DATASET,'account_alias':alias,'branch':BRANCH,'revision':revision,
        'profile':PROFILE,'upstream_revision':UPSTREAM_SHA,'model_lock':lock}
    if anatomy:
        from studio.eb01_anatomy_profile import PROFILE as ANATOMY_PROFILE,EXPERIMENT
        cfg.update(profile=ANATOMY_PROFILE,experiment=EXPERIMENT)
    source=Path('tools/kaggle_eb01_echo_parallel_entry.py').read_text();tag='__EB01_ECHO_PARALLEL_CONFIG_B64__'
    if source.count(tag)!=1:raise PilotBlocked('AMBIGUOUS_BENCHMARK_CONFIG')
    folder=root/'kernel';folder.mkdir(mode=0o700)
    (folder/'kernel_main.py').write_text(source.replace(tag,base64.b64encode(json.dumps(cfg,sort_keys=True).encode()).decode()))
    kernel=handle+'/'+slug
    (folder/'kernel-metadata.json').write_text(json.dumps({'id':kernel,'title':slug,'code_file':'kernel_main.py',
        'language':'python','kernel_type':'script','is_private':'true','enable_gpu':'true','enable_internet':'true',
        'machine_shape':'NvidiaTeslaT4','dataset_sources':[SOURCE_DATASET],
        'competition_sources':[],'kernel_sources':[],'model_sources':[]}))
    return folder,kernel


def verify_slot(folder,spec,revision):
    from studio.comfy_neural import _probe
    from studio.eb01_motion_artifact_gate import inspect_clip
    from tools.eb01_native_pixel_evidence import decoded_pixel_evidence
    from tools.eb01_echo_lossless_assembly import rgb_frame_hashes
    video=folder/'EB01_ECHO_RUSSIAN_NATIVE.mp4';result={'technical_pass':False,'actual_mp4_created':video.is_file(),
        'shot':spec['shot'],'physical_gpu_index':spec['physical_gpu_index'],'quality_approved':False}
    if not video.is_file():return result
    proof=json.loads((folder/'EB01_ECHO_ACTUAL_RECEIPT.json').read_text());digest=hashlib.sha256(video.read_bytes()).hexdigest()
    meta=_probe(video);pixels=decoded_pixel_evidence(video,frames=spec['frames'])
    qc=inspect_clip(video,expected_fps=25)
    segs=proof.get('segments',[])
    clock_ok=len(segs)==len(spec['segment_frames']);chain_ok=clock_ok
    expected_pixels = []
    offset = 0
    for i,(seg,frames) in enumerate(zip(segs,spec['segment_frames'])):
        source=folder/('SEGMENT_'+str(i)+'_NATIVE.mp4')
        tail=folder/('SEGMENT_'+str(i)+'_DECODED_TAIL.png')
        if (seg.get('index') != i or seg.get('start_scene_frame') != offset or
            seg.get('audio_conditioning_embeddings_shape') != [1, frames, 5, 12, 768] or
            seg.get('actual_native_tensor_shape')!=[1,3,frames,1024,576] or
            seg.get('native_frames')!=frames or seg.get('mp4_sha256')!=hashlib.sha256(source.read_bytes()).hexdigest() or
            seg.get('next_conditioning_png_sha256')!=hashlib.sha256(tail.read_bytes()).hexdigest()):clock_ok=False
        segment_meta=_probe(source)
        if (segment_meta['frames']!=frames or segment_meta['fps']!='25' or
            segment_meta['width']!=576 or segment_meta['height']!=1024):clock_ok=False
        source_pixels = rgb_frame_hashes(source)
        if len(source_pixels) != frames:
            clock_ok = False
        expected_pixels.extend(source_pixels[1:] if i else source_pixels)
        from PIL import Image
        with Image.open(tail) as im:
            tail_rgb = hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            if im.size != (576, 1024) or im.format != 'PNG':
                chain_ok = False
        if not source_pixels or tail_rgb != source_pixels[-1] or tail_rgb != seg.get('last_decoded_rgb_sha256'):
            chain_ok = False
        previous = segs[i - 1].get('next_conditioning_png_sha256') if i else proof.get('first_reference_sha256')
        if not previous or seg.get('conditioning_png_sha256') != previous:
            chain_ok = False
        offset += frames - 1
    audio=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','a:0','-show_entries',
        'stream=duration,sample_rate','-of','json',str(video)],timeout=45))['streams']
    checks={'native_geometry':meta['width']==576 and meta['height']==1024,
        'native_clock':meta['fps']=='25' and meta['frames']==spec['frames'],
        'no_neon':qc['status']=='NO_NEON_ARTIFACT_FOUND','all_rgb_frames_distinct':pixels['unique_decoded_rgb_frames']==spec['frames'],
        'receipt_sha':digest==proof.get('mp4_sha256'),'pinned_revision':proof.get('studio_revision')==revision,
        'pinned_upstream':proof.get('upstream_revision')==UPSTREAM_SHA,
        'profile':proof.get('profile')==spec['profile'],
        'canonical_prompt':proof.get('prompt_sha256')==spec['prompt_sha256'],
        'requested_take':all(proof.get(k) == spec[k] for k in ('shot','account_alias','seed','segment_frames')),
        'final_matches_ordered_segment_pixels': (len(expected_pixels) == spec['frames'] and
            hashlib.sha256(''.join(expected_pixels).encode()).hexdigest() == pixels['decoded_frame_hash_chain_sha256']),
        'native_steps':proof.get('steps')==spec['steps'],
        'audio_is_actual_model_input':proof.get('audio_is_model_input') is True,
        'canonical_actor':proof.get('actor')==spec['actor'] and proof.get('portrait_sha256')==spec['portrait_sha256'],
        'full_audio_embeddings':proof.get('audio_conditioning_embeddings_shape')==[1,spec['frames'],5,12,768],
        'whole_voice':proof.get('original_voice_sha256')==spec['voice_sha256'] and proof.get('original_voice_truncated') is False,
        'real_full_audio_track':len(audio)==1 and float(audio[0]['duration'])>=spec['voice_seconds'],
        'correct_physical_card':proof.get('physical_gpu_index')==spec['physical_gpu_index'],
        'real_native_segment_tensors':clock_ok,'exact_png_continuation_chain':chain_ok,
        'exact_overlap_count':proof.get('overlap_frames_discarded')==len(segs)-1,
        'text_encoder_released':proof.get('precomputed_text_encoder_released_before_inference') is True,
        'no_upscale_or_fabricated_frames':proof.get('source_video_upscaled') is False and proof.get('synthetic_frames')==0}
    if 'encoded_dialogue_clock' in proof:
        from math import isfinite
        clock = proof['encoded_dialogue_clock']
        offset_ms = clock.get('offset_ms') if isinstance(clock, dict) else None
        checks['source_waveform_within_one_frame'] = (type(offset_ms) in (float, int) and
            isfinite(offset_ms) and abs(offset_ms) <= 1000 / spec['fps'] and clock.get('status') == 'PASS')
    if 'anatomy_controls' in spec:
        from studio.eb01_anatomy_profile import native_audio_mask_numpy
        from studio.eb01_regional_motion import inspect_regions
        result['regional_motion'] = inspect_regions(video, spec['shot'])
        checks['native_anatomy_controls_applied']=proof.get('anatomy_controls')==spec['anatomy_controls']
        checks['independent_spatial_mask_bytes']=all(seg.get('spatial_audio_mask_shape')==list(native_audio_mask_numpy(spec,n).shape) and
            seg.get('spatial_audio_mask_sha256')==hashlib.sha256(native_audio_mask_numpy(spec,n).tobytes()).hexdigest()
            for seg,n in zip(segs,spec['segment_frames']))
    result.update(checks=checks,technical_pass=all(checks.values()),probe=meta,mp4_sha256=digest,
        native_inference_seconds=proof.get('native_inference_seconds'),process_wall_seconds=proof.get('process_wall_seconds'),**pixels)
    resource=proof.get('resources',{})
    result['resources']={k:resource.get(k) for k in ('per_device','host_ram_peak_gib','host_ram_total_gib',
        'process_rss_peak_gib','cpu_percent_peak','host_ram_available_gib','wall_seconds','finished')}
    return result


def run(alias,token,root,run_id,revision,attempt,lock,poll_seconds=5100,*,anatomy=False):
    from studio.eb01_anatomy_profile import PROFILE as ANATOMY_PROFILE,anatomy_specs
    report={'schema':1,'profile':PROFILE,'account_alias':alias,'run_id':run_id,
        'gpu_submission_attempts':0,'production_jobs_submitted':0,'quality_approved':False,'whole_movie_ready':False}
    if anatomy:report.update(profile=ANATOMY_PROFILE,experiment='anatomy_controls_v2',user_anatomy_feedback_resolved=False)
    try:
        auth,handle=query(token)
        if auth!='AUTHENTICATED_READ_ONLY' or not handle:raise PilotBlocked('ACCOUNT_AUTH_NOT_VERIFIED')
        folder,kernel=prepare(root,alias,handle,run_id,revision,lock,anatomy=anatomy)
        if attempt==1:
            quota=parse_quota(call(token,['kaggle','quota','--format','json'],timeout=65))
            report['official_quota_before']=quota['GPU']
            if quota['GPU']['remaining_hours']<1.25:raise PilotBlocked('INSUFFICIENT_FREE_QUOTA_FOR_BOUNDED_BENCHMARK')
            report['gpu_submission_attempts']=1
            call(token,['kaggle','kernels','push','-p',str(folder),'--timeout','3600','--accelerator','NvidiaTeslaT4'],timeout=220)
        deadline=time.monotonic()+poll_seconds
        while time.monotonic()<deadline:
            found=STATE.findall(call(token,['kaggle','kernels','status',kernel],timeout=65))
            if len(found)!=1:raise PilotBlocked('UNKNOWN_KERNEL_STATE')
            state=found[0];report['kaggle_state']=state;print('EB01_PARALLEL_STATE',alias,state,flush=True)
            if state in ('COMPLETE','ERROR','CANCELLED','CANCEL_ACKNOWLEDGED'):
                output=root/'output';output.mkdir(mode=0o700)
                call(token,['kaggle','kernels','output',kernel,'-p',str(output),'--force'],timeout=420)
                results=[]
                for slot,spec in enumerate(anatomy_specs(alias) if anatomy else specs(alias)):
                    try:results.append(verify_slot(output/('slot'+str(slot)),spec,revision))
                    except Exception as exc:results.append({'technical_pass':False,'error_type':type(exc).__name__,'slot':slot})
                report['workers']=results
                if anatomy:
                    from studio.eb01_regional_motion import compare_paired_motion
                    if len(results)==2 and all(isinstance(w.get('regional_motion'), dict) for w in results):
                        report['paired_anatomy_metric'] = compare_paired_motion(
                            results[0]['regional_motion'], results[1]['regional_motion']
                        )
                    else:
                        report['paired_anatomy_metric'] = {
                            'conclusion': 'INSUFFICIENT_EVIDENCE',
                            'human_artistic_approved': False,
                            'full_production_authorized': False,
                        }
                host=output/'EB01_ECHO_PARALLEL_HOST_SAFE.json'
                telemetry=json.loads(host.read_text()) if host.is_file() else {}
                report['host_and_both_device_measurement']={k:telemetry.get(k) for k in ('samples','host_used_peak_gib',
                    'host_available_min_gib','worker_rss_sum_peak_gib','both_gpus_compute_busy_samples','per_device','wall_seconds')}
                # Every raw MP4/PNG/checkpoint/log remains inside this envelope.
                archive=root/'private-return.zip'
                with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
                    for p in output.rglob('*'):
                        if p.is_file() and p.suffix in ('.mp4','.png','.json','.safetensors','.log'):
                            z.write(p,str(p.relative_to(output)))
                seal(archive,Path('ops/eb01_native_pilot_return_public.pem'),
                    Path(os.environ['RUNNER_TEMP'])/('EB01_PARALLEL_'+alias+'.twenc'),
                    {'purpose':'eb01-echo-parallel','run_id':run_id,'account':alias})
                passed=state=='COMPLETE' and len(results)==2 and all(x['technical_pass'] for x in results) and telemetry.get('both_gpus_compute_busy_samples',0)>0
                report['status']='TWO_SINGLE_CARD_GENERATORS_AND_CONTINUATIONS_TECHNICAL_PASS' if passed else 'PRIVATE_PARALLEL_BENCHMARK_FAILED_OR_QA_BLOCKED'
                report['encrypted_private_return_created']=True
                return report
            time.sleep(40)
        report['status']='SAME_BENCHMARK_STILL_QUEUED_OR_RUNNING_COLLECT_ONLY'
    except (PilotBlocked,ValueError,OSError,KeyError,subprocess.SubprocessError) as exc:
        report['status']='BLOCKED_'+(str(exc) if isinstance(exc,PilotBlocked) else type(exc).__name__)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--account',choices=('owner','collaborator'),required=True)
    p.add_argument('--anatomy-controls',action='store_true');a=p.parse_args()
    os.umask(0o077);lock=json.loads(base64.b64decode(os.environ['EB01_MODEL_LOCK_B64'],validate=True))
    with tempfile.TemporaryDirectory(dir=os.environ['RUNNER_TEMP'],prefix='eb01-parallel-private-') as temp:
        r=run(a.account,os.environ.get('EB01_ACCOUNT_TOKEN',''),Path(temp),os.environ['GITHUB_RUN_ID'],
              os.environ['GITHUB_SHA'],int(os.environ.get('GITHUB_RUN_ATTEMPT','1')),lock,anatomy=a.anatomy_controls)
    (Path(os.environ['RUNNER_TEMP'])/('EB01_PARALLEL_'+a.account+'_SAFE.json')).write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print('EB01_PARALLEL_NUMERICAL_RESULT',json.dumps(r,sort_keys=True))
    return 0 if r['status']=='TWO_SINGLE_CARD_GENERATORS_AND_CONTINUATIONS_TECHNICAL_PASS' else 2

if __name__=='__main__':raise SystemExit(main())
