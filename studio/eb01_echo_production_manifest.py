"""Blocked 30-job native 25 FPS production proposal; NEVER submits GPU work.

Preserves the seven-scene 40+ second story and original full voice clocks.
Do not reuse the fallback's 24 FPS / 33-frame worker measurements as Echo
capacity. Every later segment needs the previous verified decoded-tail PNG.
"""
from __future__ import annotations
import hashlib,json,math
from studio.eb01_echo_profile import PROFILE,UPSTREAM_SHA
from studio.eb01_v7_storyboard import SHOTS,SRC_SHA,VOICE_SHA,VOICE_SECONDS
from studio.eb01_v8_40s_native_motion_plan import SEGMENT_COUNTS,plan as fallback_plan

OWNER_SCENES=frozenset(('S02','S04','S06'))


def plan():
    previous={s['scene_id']:s for s in fallback_plan()['scenes']}
    scenes=[];jobs=[];cursor=0
    for scene_index,scene in enumerate(SHOTS):
        sid,actor,*_=scene;count=SEGMENT_COUNTS[sid]
        old=previous[sid]
        # Round UP to 4n+1 at the actual audio model's 25 FPS; a longer
        # interval never truncates a full original spoken line.
        length=4*math.ceil((old['unique_movie_frames']*25/24-1)/4)+1
        base=32*count+1
        extra=(length-base)//4
        if not 0<=extra<=count:raise ValueError('Unbenchmarked native segment length required')
        lengths=[37 if i<extra else 33 for i in range(count)]
        account='owner' if sid in OWNER_SCENES else 'collaborator'
        voice_frames=math.ceil(VOICE_SECONDS[sid]*25) if sid in VOICE_SECONDS else 0
        voice_start=round((length-voice_frames)/3) if voice_frames else None
        offset=0;predecessor=None;scene_jobs=[]
        for index,frames in enumerate(lengths):
            seed=2026112100+scene_index*100+index
            context={'profile':PROFILE,'upstream_sha':UPSTREAM_SHA,'scene':sid,'actor_sha256':SRC_SHA[actor],
                'voice_sha256':VOICE_SHA.get(sid),'prompt_sha256':hashlib.sha256(scene[-1].encode()).hexdigest(),
                'native_fps':25,'native_frames':frames,'start_scene_frame':offset,'seed':seed,'account':account}
            identity=hashlib.sha256(json.dumps(context,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            job_id='ECHO25_'+sid+'_'+str(index).zfill(2)+'_'+identity[:12]
            job={'job_id':job_id,'job_context_sha256':identity,'account_alias':account,'scene_id':sid,
                'segment_index':index,'native_frames':frames,'native_width':576,'native_height':1024,
                'native_fps':25,'seed':seed,'start_scene_frame':offset,
                'end_scene_frame_exclusive':offset+frames,'predecessor_job_id':predecessor,
                'first_conditioning_source':'EXACT_CANONICAL_PORTRAIT' if index==0 else 'PREDECESSOR_VERIFIED_DECODED_TAIL_PNG',
                'canonical_portrait_sha256':SRC_SHA[actor],'canonical_voice_sha256':VOICE_SHA.get(sid),
                'audio_conditioning':'SLICE_OF_FULL_SCENE_25FPS_ACOUSTIC_CONTEXT' if voice_frames else 'ZERO_AUDIO_EMBEDDINGS',
                'discard_first_overlap_frame_in_assembly':index>0,
                'require_previous_mp4_and_exported_png_hashes':index>0,
                'gpu_submitted':False,'quality_approved':False}
            jobs.append(job);scene_jobs.append(job_id);predecessor=job_id;offset+=frames-1
        assert offset+1==length
        scenes.append({'scene_id':sid,'actor':actor,'account_alias':account,'native_frames':length,
            'start_movie_frame':cursor,'end_movie_frame_exclusive':cursor+length,
            'dialogue_start_scene_frame':voice_start,
            'dialogue_end_scene_frame_exclusive':voice_start+voice_frames if voice_frames else None,
            'original_voice_seconds':VOICE_SECONDS.get(sid),'original_voice_sha256':VOICE_SHA.get(sid),
            'voice_mixed_exactly_once_per_scene':True,'job_ids':scene_jobs})
        cursor+=length
    split={a:sum(j['account_alias']==a for j in jobs) for a in ('owner','collaborator')}
    if len(jobs)!=30 or len({j['job_id'] for j in jobs})!=30 or split!={'owner':15,'collaborator':15}:
        raise ValueError('Production jobs duplicated or original balanced allocation changed')
    if cursor/25<40:raise ValueError('Native full movie cannot be shorter than approved story')
    return {'schema':1,'status':'BLOCKED_PRODUCTION_PROPOSAL_NOT_EXECUTED','engine_profile':PROFILE,
        'native_width':576,'native_height':1024,'native_fps':25,'total_original_jobs':30,
        'job_allocation':split,'generated_native_frames':sum(j['native_frames'] for j in jobs),
        'unique_final_native_frames':cursor,'native_duration_seconds':cursor/25,
        'original_voice_lines':6,'assembly_scene_order':[s['scene_id'] for s in scenes],
        'scenes':scenes,'jobs':jobs,'verified_parallel_processes_per_account':1,
        'two_simultaneous_generators_per_account_approved':False,
        'production_blockers':['TWO_ACTOR_ARTISTIC_QUALITY_CONFIRMATION_REQUIRED',
            'AUDIO_CONDITIONED_CONTINUATION_PILOT_REQUIRED','PRIVATE_EXECUTION_RECEIPT_RECHECK_REQUIRED',
            'FRESH_ACCOUNT_QUOTA_RECHECK_REQUIRED'],
        'gpu_submissions':0,'production_execution_authorized':False,'whole_movie_ready':False,
        'source_upscale_allowed':False,'synthetic_or_held_frames_allowed':False,'published':False}

if __name__=='__main__':print(json.dumps(plan(),indent=2,sort_keys=True))
