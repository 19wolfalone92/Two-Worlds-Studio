"""Join authentic native segments without a second lossy video/audio encoding.

Every assembled RGB frame must match its corresponding decoded source frame.
The overlap frame is counted once; no upscaling, interpolation or new take.
"""
from __future__ import annotations
import argparse,hashlib,json,re,subprocess,tempfile
from pathlib import Path
from studio.comfy_neural import _probe
from studio.eb01_v8_private_mux import safe_private_path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb_frame_hashes(video):
    output=subprocess.check_output(['ffmpeg','-v','error','-i',str(video),'-an','-pix_fmt','rgb24',
        '-f','framehash','-hash','sha256','-'],text=True,timeout=120)
    hashes=[]
    for line in output.splitlines():
        if not line or line.startswith('#'):continue
        value=line.rsplit(',',1)[-1].strip()
        if not re.fullmatch('[a-f0-9]{64}',value):raise ValueError('Invalid independent decoded RGB hash')
        hashes.append(value)
    return hashes


def audio_packet_hashes(video):
    doc=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','a:0',
        '-show_packets','-show_entries','packet=data_hash','-show_data_hash','sha256','-of','json',str(video)],timeout=60))
    return [p['data_hash'] for p in doc.get('packets',[])]


def join_video(segments,output):
    output=safe_private_path(Path(output))
    if output.exists():raise FileExistsError('Never overwrite original native video')
    if len(segments)<2:raise ValueError('At least two authentic native segments required')
    metas=[_probe(Path(p)) for p in segments];first=metas[0]
    if any(m['fps']!='25' or m['width']!=first['width'] or m['height']!=first['height'] for m in metas):
        raise ValueError('All source segments must share their exact native 25 FPS geometry')
    expected=[]
    for i,path in enumerate(segments):
        h=rgb_frame_hashes(path)
        if len(h)!=metas[i]['frames']:raise ValueError('Original source frame clock differs from actual pixels')
        expected.extend(h[1:] if i else h)
    output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    argv=['ffmpeg','-v','error'];filters=[];labels=[]
    for i,path in enumerate(segments):
        argv+=['-i',str(path)];label='v'+str(i)
        filters.append(f'[{i}:v]'+('trim=start_frame=1,' if i else '')+f'setpts=PTS-STARTPTS[{label}]')
        labels.append('['+label+']')
    filters.append(''.join(labels)+f'concat=n={len(segments)}:v=1:a=0[video]')
    subprocess.run(argv+['-filter_complex',';'.join(filters),'-map','[video]','-c:v','libx264',
        '-crf','0','-preset','medium','-pix_fmt','yuv420p','-video_track_timescale','25',str(output)],
        check=True,capture_output=True,timeout=240)
    actual=rgb_frame_hashes(output)
    checks={'every_native_rgb_frame_identical':actual==expected,
        'exact_native_frames':_probe(output)['frames']==len(expected),
        'all_final_rgb_frames_distinct':len(set(actual))==len(expected)}
    if not all(checks.values()):raise ValueError('Native video join changed decoded source pixels')
    return {'checks':checks,'source_segment_sha256':[sha(p) for p in segments],
        'probe':_probe(output),'native_rgb_hash_chain_sha256':hashlib.sha256(''.join(actual).encode()).hexdigest()}


def assemble(segments,audio_source,output):
    output=safe_private_path(Path(output))
    if output.exists():raise FileExistsError('Never overwrite original native video')
    output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='lossless-native-join-') as temp:
        silent=Path(temp)/'silent.mp4';joined=join_video(segments,silent)
        n=joined['probe']['frames']
        subprocess.run(['ffmpeg','-v','error','-i',str(silent),'-i',str(audio_source),'-map','0:v:0',
            '-map','1:a:0','-c','copy','-t',str(n/25),'-video_track_timescale','25',
            '-movflags','+faststart',str(output)],check=True,capture_output=True,timeout=90)
    actual=rgb_frame_hashes(output);original_audio=audio_packet_hashes(audio_source)
    checks={'every_native_rgb_frame_identical':joined['native_rgb_hash_chain_sha256']==hashlib.sha256(''.join(actual).encode()).hexdigest(),
        'all_aac_packet_bytes_identical':bool(original_audio) and original_audio==audio_packet_hashes(output),
        'exact_native_frames':_probe(output)['frames']==n,
        'all_final_rgb_frames_distinct':len(set(actual))==n}
    proof={'schema':1,'status':'LOSSLESS_NATIVE_ASSEMBLY_TECHNICAL_PASS' if all(checks.values()) else 'ASSEMBLY_QA_BLOCKED',
        'checks':checks,'source_segment_sha256':[sha(p) for p in segments],
        'source_audio_mp4_sha256':sha(audio_source),'mp4_sha256':sha(output),
        'probe':_probe(output),'native_rgb_hash_chain_sha256':hashlib.sha256(''.join(actual).encode()).hexdigest(),
        'overlap_frames_discarded':len(segments)-1,'video_encoding':'H264_LOSSLESS_CRF0',
        'audio_encoding':'ORIGINAL_AAC_PACKETS_COPIED','second_lossy_compression':False,
        'new_diffusion_steps':0,'new_original_takes':0,'source_video_upscaled':False,
        'synthetic_frames':0,'human_quality_approved':False,'whole_movie_ready':False}
    output.with_suffix('.receipt.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
    if not all(checks.values()):raise ValueError('Lossless assembly failed exact source RGB/audio equality')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--segment',action='append',type=Path,required=True)
    p.add_argument('--audio-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(assemble(a.segment,a.audio_source,a.output),sort_keys=True))
