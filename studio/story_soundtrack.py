"""Frame-aligned Russian narration mastering and a changing original soundtrack.

No speech acceleration, source edits, paid APIs or publication. Original scene
WAVs are preserved; all mastering is written to new files with measured timing.
"""
from __future__ import annotations
import json
import math
import os
import subprocess
from pathlib import Path
from .shorts_factory import duration,sha,frame_count
from .autonomous_factory import inventory,mood_of,atomic_json
from .music_score import Piano,compose_score


def ffmpeg(args,output: Path) -> str:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_name('.'+output.stem+'.building'+output.suffix)
    result=subprocess.run(['ffmpeg','-hide_banner','-nostdin','-nostats','-y',*args,str(temp)],
                          capture_output=True,text=True,timeout=240)
    try:
        if result.returncode:
            raise RuntimeError(result.stderr[-2200:])
        os.link(temp,output)
    finally:
        temp.unlink(missing_ok=True)
    return result.stderr


def normalize_voice(source: Path,dest: Path,seconds: float) -> dict:
    receipt=dest.with_suffix('.receipt.json')
    if dest.exists():
        if receipt.exists() and abs(duration(dest)-seconds)<.001:
            saved=json.loads(receipt.read_text())
            if saved.get('source_sha256')!=sha(source) or saved.get('sha256')!=sha(dest):
                raise ValueError('Mastered voice source or cached WAV changed; rebuild in a new destination')
            return saved
        raise ValueError('Existing mastered voice lacks a matching measured receipt')
    raw_length=duration(source)
    if seconds+1e-6<raw_length or seconds-raw_length>.05:
        raise ValueError('Voice timing may only be padded to the next video frame')
    log=ffmpeg(['-i',str(source),'-af',
        'loudnorm=I=-18:LRA=11:TP=-2:print_format=json,aresample=48000,'
        f'apad,atrim=end_sample={round(seconds*48000)}',
        '-ar','48000','-ac','1','-c:a','pcm_s16le'],dest)
    start,end=log.rfind('{'),log.rfind('}')
    loudness=json.loads(log[start:end+1])
    if not math.isfinite(float(loudness['output_i'])) or abs(duration(dest)-seconds)>.001:
        raise ValueError('Mastered narration is silent or not aligned to the original video clock')
    result={'source_sha256':sha(source),'sha256':sha(dest),'source_seconds':raw_length,
            'master_seconds':duration(dest),'speech_rate_changed':False,'loudness':loudness,
            'listening_review':'pending'}
    atomic_json(receipt,result)
    return result


def ambience_for(story: str,scene: int) -> str:
    if story=='13':return 'clock_shop' if scene in [1,4,8] else 'quiet_room'
    if story=='14':return 'domestic_room'
    if story=='15':return 'rain_coach' if scene<8 else 'quiet_room'
    if story=='16':return 'greenhouse' if scene==2 else 'domestic_room' if scene==8 else 'habitat_fan'
    if story=='17':return 'quiet_room' if scene==2 else 'domestic_room'
    return 'quiet_room'


def _assembly_cache(dest: Path, sources: list[Path], lengths: list[float], **settings) -> tuple[bool, dict]:
    fingerprint={'schema':1,'input_sha256':[sha(p) for p in sources],
                 'seconds':lengths,'settings':settings}
    receipt=dest.with_suffix('.assembly.json')
    if dest.exists():
        saved=json.loads(receipt.read_text()) if receipt.is_file() else {}
        if saved.get('inputs')!=fingerprint or saved.get('sha256')!=sha(dest):
            raise ValueError('Assembled audio source or cached WAV changed; rebuild in a new destination')
        return True,fingerprint
    return False,fingerprint


def concat_voices(sources: list[Path], lengths: list[float], dest: Path) -> None:
    if len(sources)!=len(lengths) or not sources:raise ValueError('Voice assembly needs matching inputs')
    cached,inputs=_assembly_cache(dest,sources,lengths,operation='voice_concat')
    if cached:return
    playlist=dest.with_suffix('.concat.txt')
    playlist.parent.mkdir(parents=True,exist_ok=True)
    playlist.write_text(''.join(f"file '{v.resolve()}'\n" for v in sources))
    try:
        ffmpeg(['-f','concat','-safe','0','-i',str(playlist),'-c:a','pcm_s16le'],dest)
    finally:
        playlist.unlink(missing_ok=True)
    atomic_json(dest.with_suffix('.assembly.json'),{'inputs':inputs,'sha256':sha(dest)})


def crossfade_beds(sources: list[Path],lengths: list[float],dest: Path,*,fade: float,loop=False):
    if len(sources)!=len(lengths) or not sources:raise ValueError('Bed assembly needs matching inputs')
    cached,inputs=_assembly_cache(dest,sources,lengths,operation='bed_crossfade',fade=fade,loop=loop)
    if cached:return ''
    args=[];filters=[]
    for i,(source,seconds) in enumerate(zip(sources,lengths)):
        if loop:args+=['-stream_loop','-1']
        args+=['-i',str(source)]
        filters.append(f'[{i}:a]aresample=48000,aformat=channel_layouts=stereo,'
                       f'apad,atrim=duration={seconds+fade:.9f},asetpts=PTS-STARTPTS[a{i}]')
    prev='a0'
    for i in range(1,len(sources)):
        name=f'join{i}'
        filters.append(f'[{prev}][a{i}]acrossfade=d={fade}:c1=tri:c2=tri[{name}]')
        prev=name
    filters.append(f'[{prev}]atrim=duration={sum(lengths):.9f}[out]')
    log=ffmpeg([*args,'-filter_complex',';'.join(filters),'-map','[out]',
                '-ar','48000','-ac','2','-c:a','pcm_s16le'],dest)
    atomic_json(dest.with_suffix('.assembly.json'),{'inputs':inputs,'sha256':sha(dest)})
    return log


def mix_tracks(voice: Path,music: Path,ambience: Path,dest: Path,seconds: float) -> dict:
    sources={'voice':sha(voice),'music':sha(music),'ambience':sha(ambience)}
    receipt=dest.with_suffix('.receipt.json')
    if dest.exists():
        saved=json.loads(receipt.read_text()) if receipt.is_file() else {}
        if (saved.get('source_sha256')==sources and saved.get('sha256')==sha(dest)
                and abs(saved.get('seconds',-1)-seconds)<.001):
            return saved
        raise ValueError('Mix source or cached WAV changed; rebuild in a new destination')
    log=ffmpeg(['-i',str(voice),'-i',str(music),'-i',str(ambience),'-filter_complex',
        '[0:a]aresample=48000,aformat=channel_layouts=stereo,asplit=2[v][key];'
        '[1:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.8[m];'
        '[2:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.8[e];'
        '[m][e]amix=inputs=2:normalize=0:duration=longest[bed];'
        '[bed][key]sidechaincompress=threshold=0.05:ratio=4:attack=20:release=400[ducked];'
        '[v][ducked]amix=inputs=2:normalize=0:duration=first,'
        'loudnorm=I=-16:LRA=11:TP=-1.5:print_format=json,aresample=48000,'
        f'apad,atrim=end_sample={round(seconds*48000)},asetpts=N/SR/TB[out]',
        '-map','[out]','-ar','48000','-ac','2','-c:a','pcm_s16le'],dest)
    if abs(duration(dest)-seconds)>.001:
        raise ValueError('Mixed audio does not preserve the full narration timeline')
    result={'sha256':sha(dest),'source_sha256':sources,'seconds':duration(dest),'speech_rate_changed':False,
            'automatic_music_ducking':True,
            'loudness':json.loads(log[log.rfind('{'):log.rfind('}')+1]),'listening_review':'pending'}
    atomic_json(dest.with_suffix('.receipt.json'),result)
    return result


def prepare_story(project: Path,piano_path: Path,sound_bank: Path) -> dict:
    root=project.parent
    state=inventory(root)
    if state['missing']['voice']:
        raise ValueError('Full original narration must exist before mastering a story')
    doc=json.loads(project.read_text());story=doc['id'].replace('work_','')
    shots={x['scene']:x for x in json.loads((root/'shot_plan.json').read_text())}
    piano=Piano(piano_path);lengths=[];voices=[];scores=[];ambiences=[];receipts=[]
    for scene in doc['scenes']:
        sid=scene['id'];source=root/scene['voice']
        expected=' '.join(shots[sid]['exact_source_text'].split())
        spoken=' '.join(' '.join(c['text'].split()) for c in scene['subtitles'])
        if expected!=spoken:
            raise ValueError('Actual subtitle transcript omits or alters the locked scene text')
        seconds=frame_count(duration(source),doc['fps'])/doc['fps'];lengths.append(seconds)
        normalized=root/f'voices/master_scene_{sid:02d}.wav'
        receipts.append(normalize_voice(source,normalized,seconds));voices.append(normalized)
        score=root/scene.get('music',f'music/scene_{sid:02d}_piano.wav')
        if not score.exists():
            info=compose_score(score,seconds+1,piano,mood=mood_of(shots[sid]),seed=int(story)*100+sid)
            atomic_json(score.with_suffix('.receipt.json'),info)
        scene['music']=str(score.relative_to(root))
        scores.append(score)
        ambience=sound_bank/(ambience_for(story,sid)+'.wav')
        if not ambience.exists():raise FileNotFoundError(ambience)
        ambiences.append(ambience)
    audio=root/'audio';audio.mkdir(exist_ok=True)
    narration=audio/'narration_ru.wav';music=audio/'original_score.wav';ambience=audio/'environment.wav';mix=audio/'story_mix.wav'
    concat_voices(voices,lengths,narration)
    crossfade_beds(scores,lengths,music,fade=1.0)
    crossfade_beds(ambiences,lengths,ambience,fade=.4,loop=True)
    result=mix_tracks(narration,music,ambience,mix,sum(lengths))
    doc['master_audio']=str(mix.relative_to(root));atomic_json(project,doc)
    report={'story':story,'script_sha256':doc['script_sha256'],'seconds':sum(lengths),
            'all_original_scene_words_preserved':True,'voices':receipts,'mix':result,
            'music':'original scene compositions, public-domain piano samples','music_crossfade_seconds':1.0,
            'ambience':'original procedural environmental sound design','published':False}
    atomic_json(audio/'audio_qa.json',report)
    return report
