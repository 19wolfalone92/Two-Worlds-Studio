"""Independent CPU lip-sync measurement on a manually reviewed HUMAN face crop.

Analysis-only crops and deliberately shifted negative controls stay private.
They are never returned as native renders and never authorize artistic release.
"""
from __future__ import annotations
import argparse,hashlib,json,math,subprocess,sys,time
from pathlib import Path

UPSTREAM_SHA='907c0b579c2e2d83f0eae1b2ac9e720cde4e5623'
WEIGHTS_SHA='961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def probe(path):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-count_frames','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames',
        '-of','json',str(path)],timeout=45))['streams'][0]
    if data['avg_frame_rate']!='25/1':raise ValueError('SyncNet requires original 25 FPS; never fabricate native frames')
    return {k:int(data[k]) for k in ('width','height','nb_read_frames')}


def review(video,runtime,checkpoint,crop,seconds,out):
    import torch
    if not 1<=seconds<=8 or not math.isfinite(seconds):raise ValueError('Invalid original speech duration')
    if out.exists():raise FileExistsError('Never overwrite original analysis or native footage')
    native=probe(video);x,y,size=crop
    if (min(x,y)<0 or size<96 or x+size>native['width'] or y+size>native['height']):
        raise ValueError('Manually reviewed face crop lies outside native frame')
    revision=subprocess.check_output(['git','-C',str(runtime),'rev-parse','HEAD'],text=True).strip()
    if revision!=UPSTREAM_SHA or sha(checkpoint)!=WEIGHTS_SHA:raise ValueError('Unreviewed SyncNet source or weights')
    sys.path.insert(0,str(runtime))
    from SyncNetInstance import SyncNetInstance
    torch.set_num_threads(2)
    model=SyncNetInstance(device='cpu')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.__S__.load_state_dict(state,strict=True)
    out.mkdir(parents=True)
    count=min(native['nb_read_frames'],round(seconds*25))
    if abs(count/25-seconds)>.021:raise ValueError('Original speech does not fit a complete native 25 FPS clock')
    results=[];began=time.monotonic()
    for delay_ms in (0,200):
        analysis=out/('ANALYSIS_ONLY_AUDIO_DELAY_'+str(delay_ms)+'.avi')
        argv=['ffmpeg','-v','error','-i',str(video),'-map','0:v:0','-map','0:a:0',
            '-vf',f'crop={size}:{size}:{x}:{y},scale=224:224:flags=lanczos',
            '-frames:v',str(count),'-t',str(count/25),'-c:v','mjpeg','-q:v','2',
            '-c:a','pcm_s16le','-ar','16000','-ac','1']
        if delay_ms:argv+=['-af',f'adelay={delay_ms},apad']
        subprocess.run(argv+[str(analysis)],check=True,capture_output=True,timeout=90)
        if probe(analysis)['nb_read_frames']!=count:raise ValueError('Analysis conversion changed native frame clock')
        opt=argparse.Namespace(tmp_dir=str(out/'private-model-work'),reference='delay'+str(delay_ms),batch_size=8,vshift=15)
        with torch.inference_mode():offset,confidence,dists=model.evaluate(opt,str(analysis))
        results.append({'deliberate_analysis_audio_delay_ms':delay_ms,'offset_frames':int(offset),
            'offset_milliseconds':int(offset)*40,'confidence':round(float(confidence),6),
            'minimum_distance':round(float(dists.mean(axis=0).min()),6)})
    difference=abs(results[1]['offset_frames']-results[0]['offset_frames'])
    result={'schema':1,'status':'HUMAN_FACE_AUDIOVISUAL_OFFSET_MEASURED_REVIEW_REQUIRED',
        'source_original_mp4_sha256':sha(video),'native_video':native,'native_fps':25,
        'analysis_speech_seconds':count/25,'manual_face_crop_xy_size':list(crop),
        'syncnet_upstream_sha':UPSTREAM_SHA,'syncnet_weights_sha256':WEIGHTS_SHA,
        'original_measurement':results[0],'negative_control':results[1],
        'negative_control_offset_change_frames':difference,
        'negative_control_response_verified':abs(difference-5)<=1,
        'cpu_only':True,'wall_seconds':round(time.monotonic()-began,3),
        'native_source_modified':False,'human_artistic_approved':False,
        'human_lipsync_approved':False,'skeleton_metric_validated':False,'whole_movie_approved':False}
    (out/'INDEPENDENT_HUMAN_LIPSYNC_SAFE.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--face-crop',type=int,nargs=3,required=True,metavar=('X','Y','SIZE'))
    p.add_argument('--speech-seconds',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(review(args.video,args.runtime,args.checkpoint,args.face_crop,args.speech_seconds,args.output),sort_keys=True))

if __name__=='__main__':main()
