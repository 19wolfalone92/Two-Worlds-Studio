"""Measured editorial assembly of real neural shots and photographic inserts.

Short generated takes may be reused at different framings. The receipt reports
unique neural footage separately from its screen time; it never calls a crop
or a held photograph character animation.
"""
from __future__ import annotations
import math
import subprocess
from pathlib import Path


def shot_plan(seconds: float,clip_seconds: list[float],fps: int) -> list[dict]:
    if not math.isfinite(seconds) or seconds<=0 or fps<1 or not clip_seconds:
        raise ValueError('An actual narration clock and neural takes are required')
    total=math.ceil(seconds*fps)
    lengths=[max(1,int(round(x*fps))) for x in clip_seconds]
    if any(not math.isfinite(x) or x<=0 for x in clip_seconds):raise ValueError('Invalid real take duration')
    # Keep every take intact when possible. Reuse is explicit, never hidden as
    # newly generated footage. A photographic pause separates each take group.
    cycle_frames=sum(lengths)
    cycles=max(1,min(5,math.ceil(total*.64/cycle_frames)))
    while cycles>1 and cycles*cycle_frames>total-fps:cycles-=1
    if cycle_frames>=total:
        return [{'kind':'motion','clip':0,'frames':total,'zoom':1.0,'cx':.5,'cy':.5}]
    remaining=total-cycles*cycle_frames
    inserts=[remaining//cycles+(i<remaining%cycles) for i in range(cycles)]
    shots=[]
    for cycle in range(cycles):
        for index,length in enumerate(lengths):
            zoom=[1.0,1.14,1.35,1.2,1.0][(cycle+index)%5]
            shots.append({'kind':'motion','clip':index,'frames':length,'zoom':zoom,'cx':.5,'cy':.43 if zoom>1 else .5})
        if inserts[cycle]:
            shots.append({'kind':'photo','frames':inserts[cycle],
                          'zoom':[1.0,1.28,1.12,1.4,1.0][cycle],'cx':.5,'cy':.42 if cycle else .5})
    if sum(s['frames'] for s in shots)!=total:raise AssertionError('Editorial clock mismatch')
    return shots


def crop_filter(shot: dict,w: int,h: int) -> str:
    zoom=shot['zoom'];fraction=1/zoom
    x=max(0,min(1-fraction,shot['cx']-fraction/2))
    y=max(0,min(1-fraction,shot['cy']-fraction/2))
    return (f'crop=w=trunc(iw/{zoom}/2)*2:h=trunc(ih/{zoom}/2)*2:'
            f'x=trunc(iw*{x:.8f}/2)*2:y=trunc(ih*{y:.8f}/2)*2,'
            f'scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos:out_color_matrix=bt709:out_range=tv,crop={w}:{h},setsar=1')


def render(scene: dict,output: Path,w: int,h: int,fps: int,subtitles: Path | None) -> dict:
    from .shorts_factory import video_duration,sha
    paths=scene['motion_clips'];lengths=[video_duration(p) for p in paths]
    shots=shot_plan(scene['seconds'],lengths,fps)
    args=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y',
          '-filter_complex_threads','1']
    filters=[]
    for index,shot in enumerate(shots):
        if shot['kind']=='motion':args+=['-threads','1','-i',str(paths[shot['clip']])]
        else:args+=['-loop','1','-framerate',str(fps),'-i',str(scene['image'])]
        filters.append(f'[{index}:v]fps={fps},{crop_filter(shot,w,h)},'
            f'tpad=stop_mode=clone:stop_duration={shot["frames"]/fps:.9f},'
            f'trim=end_frame={shot["frames"]},setpts=N/{fps}/TB,format=yuv420p[s{index}]')
    voice_index=len(shots);args+=['-i',str(scene['voice'])]
    filters.append(''.join(f'[s{i}]' for i in range(len(shots)))+f'concat=n={len(shots)}:v=1:a=0[edit]')
    vf='[edit]null'
    if subtitles:
        escaped=str(subtitles).replace('\\','\\\\').replace(':','\\:').replace("'","\\'")
        vf=f"[edit]ass='{escaped}'"
    filters.append(vf+'[v]')
    filters.append(f'[{voice_index}:a]aresample=48000,aformat=channel_layouts=stereo,apad[voice]')
    if scene.get('music'):
        args+=['-stream_loop','-1','-i',str(scene['music'])]
        filters.append(f'[voice]asplit=2[main][key];[{voice_index+1}:a]aresample=48000,'
            'aformat=channel_layouts=stereo,volume=0.22[bed];'
            '[bed][key]sidechaincompress=threshold=0.025:ratio=8:attack=30:release=350[duck];'
            '[main][duck]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.92[a]')
    else:filters.append('[voice]alimiter=limit=0.92[a]')
    frames=math.ceil(scene['seconds']*fps)
    args+=['-filter_complex',';'.join(filters),'-map','[v]','-map','[a]',
        '-frames:v',str(frames),'-t',f'{frames/fps:.9f}','-c:v','libx264','-preset','medium',
        '-crf','19','-threads','3','-pix_fmt','yuv420p','-r',str(fps),'-c:a','aac',
        '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
        '-b:a','192k','-ar','48000','-ac','2','-movflags','+faststart',str(output)]
    result=subprocess.run(args,capture_output=True,text=True,timeout=1800)
    if result.returncode:raise RuntimeError('Real scene edit failed: '+result.stderr[-2400:])
    if abs(video_duration(output)-frames/fps)>.08:raise RuntimeError('Rendered scene lost narration frames')
    unique={sha(p):length for p,length in zip(paths,lengths)}
    return {'unique_neural_input_seconds':sum(unique.values()),
        'neural_shot_screen_seconds':sum(s['frames']/fps for s in shots if s['kind']=='motion'),
        'photographic_insert_seconds':sum(s['frames']/fps for s in shots if s['kind']=='photo'),
        'takes_reused':True,'photographs_are_not_character_animation':True,
        'shots':shots,'input_sha256':{str(p):sha(p) for p in paths}}
