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
    if any(not math.isfinite(x) or x<=0 for x in clip_seconds):raise ValueError('Invalid real take duration')
    from .shorts_factory import frame_count
    total=frame_count(seconds,fps)
    # Use only complete source frames. Rounding a 1.49-frame input up to two
    # creates an unrequested held frame at the end of a take.
    lengths=[max(1,int(math.floor(x*fps+.0001))) for x in clip_seconds]
    # Keep every take intact when possible. Reuse is explicit, never hidden as
    # newly generated footage. A photographic pause separates each take group.
    cycle_frames=sum(lengths)
    cycles=max(1,min(5,math.ceil(total*.64/cycle_frames)))
    while cycles>1 and cycles*cycle_frames>total-fps:cycles-=1
    if cycle_frames>=total:
        # Honor all supplied takes in order. Previously this assigned the full
        # narration duration to take zero and froze it while ignoring take one.
        shots=[];remaining=total
        for index,length in enumerate(lengths):
            count=min(length,remaining)
            if count:shots.append({'kind':'motion','clip':index,'frames':count,'zoom':1.0,'cx':.5,'cy':.5})
            remaining-=count
            if not remaining:break
        return shots
    remaining=total-cycles*cycle_frames
    inserts=[remaining//cycles+(i<remaining%cycles) for i in range(cycles)]
    shots=[]
    for cycle in range(cycles):
        for index,length in enumerate(lengths):
            zoom=[1.0,1.025,1.04,1.025,1.0][(cycle+index)%5]
            shots.append({'kind':'motion','clip':index,'frames':length,'zoom':zoom,'cx':.5,'cy':.43 if zoom>1 else .5})
        if inserts[cycle]:
            shots.append({'kind':'photo','frames':inserts[cycle],
                          'zoom':[1.0,1.025,1.04,1.025,1.0][cycle],'cx':.5,'cy':.46 if cycle else .5})
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
    from .shorts_factory import video_duration,sha,frame_count
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
    next_input=len(shots)
    focus=scene.get('foreground_soft_focus')
    if focus:
        from PIL import Image,ImageDraw
        mask=output.with_suffix('.focus-mask.png');matte=Image.new('L',(w,h));draw=ImageDraw.Draw(matte)
        for row in range(h):
            value=max(0,min(1,(row/h-focus['start'])/(focus['end']-focus['start'])))
            value=value*value*(3-2*value)
            draw.line((0,row,w,row),fill=round(255*value))
        matte.save(mask)
        mask_index=next_input;next_input+=1
        args+=['-loop','1','-framerate',str(fps),'-i',str(mask)]
    voice_index=next_input;args+=['-i',str(scene['voice'])]
    filters.append(''.join(f'[s{i}]' for i in range(len(shots)))+f'concat=n={len(shots)}:v=1:a=0[edit]')
    name='edit'
    if focus:
        filters.append(f"[edit]split=2[sharp][softsource];[softsource]gblur=sigma={focus['sigma']}[soft];"
            f'[{mask_index}:v]format=yuv420p[mask];[sharp][soft][mask]maskedmerge=planes=1[focused]')
        name='focused'
    vf=f'[{name}]null'
    if subtitles:
        escaped=str(subtitles).replace('\\','\\\\').replace(':','\\:').replace("'","\\'")
        vf=f"[{name}]ass='{escaped}'"
    filters.append(vf+'[v]')
    filters.append(f'[{voice_index}:a]aresample=48000,aformat=channel_layouts=stereo,apad[voice]')
    if scene.get('music'):
        args+=['-stream_loop','-1','-i',str(scene['music'])]
        filters.append(f'[voice]asplit=2[main][key];[{voice_index+1}:a]aresample=48000,'
            'aformat=channel_layouts=stereo,volume=0.22[bed];'
            '[bed][key]sidechaincompress=threshold=0.025:ratio=8:attack=30:release=350[duck];'
            '[main][duck]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.92:latency=1[a]')
    else:filters.append('[voice]alimiter=limit=0.92:latency=1[a]')
    frames=frame_count(scene['seconds'],fps)
    args+=['-filter_complex',';'.join(filters),'-map','[v]','-map','[a]',
        '-frames:v',str(frames),'-t',f'{frames/fps:.9f}','-c:v','libx264','-preset','veryfast',
        '-progress',str(output.with_suffix('.progress.txt')),'-stats_period','5',
        '-crf','18','-threads','3','-pix_fmt','yuv420p','-r',str(fps),'-c:a',
        'pcm_s16le' if output.suffix == '.mov' else 'aac',
        '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
        '-b:a','192k','-ar','48000','-ac','2','-movflags','+faststart',str(output)]
    result=subprocess.run(args,capture_output=True,text=True,timeout=1800)
    if result.returncode:raise RuntimeError('Real scene edit failed: '+result.stderr[-2400:])
    if abs(video_duration(output)-frames/fps)>.08:raise RuntimeError('Rendered scene lost narration frames')
    unique={sha(p):length for p,length in zip(paths,lengths)}
    used=[sha(paths[s['clip']]) for s in shots if s['kind']=='motion']
    return {'unique_neural_input_seconds':sum(unique.values()),
        'neural_shot_screen_seconds':sum(s['frames']/fps for s in shots if s['kind']=='motion'),
        'photographic_insert_seconds':sum(s['frames']/fps for s in shots if s['kind']=='photo'),
        'takes_reused':len(used)>len(set(used)),
        'photographs_are_not_character_animation':True,
        'shots':shots,'input_sha256':{str(p):sha(p) for p in paths}}
