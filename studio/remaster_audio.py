"""Scene-clock narration and sampled scores for photographic story remasters.

Consumes existing WAVs and an owned instrument bank. No text generation,
API calls, voice cloning, time stretching, publishing, or review claims.
"""
from __future__ import annotations
import json,math,re,subprocess,wave
from pathlib import Path
import numpy as np

SR=48000

def read_voice(path:Path)->tuple[np.ndarray,int]:
    with wave.open(str(path),'rb') as f:
        if f.getsampwidth()!=2 or f.getnchannels()!=1:
            raise ValueError('Expected a mono PCM16 narration take')
        return np.frombuffer(f.readframes(f.getnframes()),'<i2').astype(np.float32)/32768,f.getframerate()

def write_wave(path:Path,samples:np.ndarray,rate:int=SR):
    path.parent.mkdir(parents=True,exist_ok=True)
    samples=np.asarray(samples)
    if not len(samples) or not np.isfinite(samples).all():raise ValueError('Empty or non-finite audio')
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1 if samples.ndim==1 else samples.shape[1]);f.setsampwidth(2);f.setframerate(rate)
        f.writeframes((np.clip(samples,-.9999,.9999)*32767).astype('<i2').tobytes())

def scene_wave(path:Path,words:list[dict],*,first:bool=False,last:bool=False,
               opening_s:float=.06)->tuple[np.ndarray,list[dict],dict]:
    """Remove only exterior silence; keep the entire spoken span at native speed."""
    raw,rate=read_voice(path)
    if not words:raise ValueError('Measured word alignment is required')
    speech0=min(w['start'] for w in words);speech1=max(w['end'] for w in words)
    active=np.flatnonzero(np.abs(raw)>.0010)
    if not len(active):raise ValueError('Silent narration')
    begin=max(0,min(round((speech0-.18)*rate),int(active[0]-.1*rate)))
    end=min(len(raw),max(round((speech1+.22)*rate),int(active[-1]+.12*rate)))
    if begin>speech0*rate or end<speech1*rate:raise ValueError('Cut would remove aligned speech')
    take=raw[begin:end]
    # Exact sample-rate conversion changes sampling density, never speech speed.
    if rate!=SR:
        count=round(len(take)*SR/rate)
        take=np.interp(np.arange(count)*rate/SR,np.arange(len(take)),take).astype(np.float32)
    lead=opening_s if first else .035
    tail=.70 if last else .12
    count=math.ceil((lead+len(take)/SR+tail)*30)*int(SR/30)
    result=np.zeros(count,dtype=np.float32);offset=round(lead*SR)
    result[offset:offset+len(take)]=take
    shift=offset/SR-begin/rate
    adjusted=[{**w,'start':round(w['start']+shift,4),'end':round(w['end']+shift,4)} for w in words]
    return result,adjusted,{'removed_head_s':begin/rate,'removed_tail_s':(len(raw)-end)/rate,
                          'duration_s':len(result)/SR,'speech_speed_factor':1.0}

def normalize(source:Path,dest:Path,*,lufs:float,tp:float=-1.7)->dict:
    base=['ffmpeg','-v','info','-nostdin','-i',str(source),'-af',f'loudnorm=I={lufs}:LRA=9:TP={tp}:print_format=json','-f','null','-']
    first=subprocess.run(base,capture_output=True,text=True,check=True)
    match=re.findall(r'\{\s*"input_i".*?\}',first.stderr,re.S)
    if not match:raise RuntimeError('Loudness measurement unavailable')
    measured=json.loads(match[-1])
    filt=(f'loudnorm=I={lufs}:LRA=9:TP={tp}:measured_I={measured["input_i"]}:'
          f'measured_LRA={measured["input_lra"]}:measured_TP={measured["input_tp"]}:'
          f'measured_thresh={measured["input_thresh"]}:offset={measured["target_offset"]}:linear=true')
    subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(source),'-af',filt,
                    '-ar',str(SR),'-c:a','pcm_s16le',str(dest)],check=True)
    return measured

def sampled_score(bank:Path,destination:Path,seconds:float,index:int,scene_starts:list[float])->dict:
    """Re-score the studio's existing six harmonic palettes for the new timeline."""
    import tinysoundfont
    chords=[[[50,57,60,64],[46,53,57,60],[48,55,60,64],[45,52,57,61]],
            [[53,60,63,68],[49,56,60,65],[56,63,67,70],[51,58,63,65]],
            [[40,47,55,59],[36,43,52,55],[35,42,54,59],[40,47,52,59]],
            [[47,54,57,61],[43,50,54,59],[50,57,61,66],[45,52,59,61]],
            [[48,55,59,64],[45,52,55,60],[41,48,52,60],[43,50,55,62]],
            [[43,50,53,58],[39,46,50,55],[46,53,57,62],[38,45,50,57]]][index]
    bpm=[62,58,56,60,60,62][index];beat=60/bpm
    rng=np.random.default_rng(70700+index);events=[];notes=[]
    def note(t,channel,pitch,length,velocity):
        if t>=seconds-.4:return
        stop=min(seconds-.2,t+length)
        if stop<=t:return
        events.extend([(max(0,t),True,channel,int(pitch),int(velocity)),(stop,False,channel,int(pitch),0)])
        notes.append({'start':round(t,3),'end':round(stop,3),'note':int(pitch),'channel':channel})
    for bar in range(math.ceil(seconds/(4*beat))):
        t=bar*4*beat;chord=chords[bar%len(chords)]
        progress=t/seconds;arc=.55+.35*math.sin(math.pi*progress)
        if index==2 and progress<.45:
            note(t,0,chord[0]-12,beat*3.2,27)
            if bar%2==1:note(t+2*beat,0,chord[2]+12,beat*2,23)
            continue
        note(t,0,max(28,chord[0]-12),beat*3.6,round(36*arc))
        for j,p in enumerate([1,2,3,2]):
            note(t+beat*j+rng.uniform(0,.018),0,chord[p],beat*2.2,round(rng.uniform(30,44)*arc))
        if progress>.25:
            for pitch in chord[1:]:note(t+.05,1,pitch,beat*3.7,round(18*arc))
    # One sparse melodic response after the story changes meaning.
    turn=scene_starts[min(5,len(scene_starts)-1)]
    if index!=2:
        for j,pitch in enumerate([chords[2][2]+12,chords[2][1]+12,chords[2][3]+12]):
            note(turn+j*1.5*beat,2,pitch,beat*2,25)
    synth=tinysoundfont.Synth(gain=-9,samplerate=SR);sfid=synth.sfload(str(bank))
    for channel,program,volume,pan in [(0,2,90,53),(1,[40,24,97,97,24,40][index],30,75),(2,2,53,80)]:
        synth.program_select(channel,sfid,0,program);synth.control_change(channel,7,volume);synth.control_change(channel,10,pan)
    blocks=[];cursor=0;total=round(seconds*SR)
    for t,on,channel,pitch,velocity in sorted(events,key=lambda e:(e[0],e[1])):
        pos=min(total,round(t*SR))
        if pos>cursor:blocks.append(np.frombuffer(synth.generate_simple(pos-cursor),np.float32).reshape(-1,2).copy());cursor=pos
        if on:synth.noteon(channel,pitch,velocity)
        else:synth.noteoff(channel,pitch)
    if cursor<total:blocks.append(np.frombuffer(synth.generate_simple(total-cursor),np.float32).reshape(-1,2).copy())
    music=np.concatenate(blocks)[:total];dry=music.copy()
    for delay,gain in [(.079,.12),(.131,.08),(.277,.05),(.433,.035)]:
        n=round(delay*SR);music[n:,0]+=dry[:-n,1]*gain;music[n:,1]+=dry[:-n,0]*gain
    fade=min(round(.8*SR),total);music[:fade]*=np.linspace(0,1,fade)[:,None]
    fade=min(round(2.2*SR),total);music[-fade:]*=np.linspace(1,0,fade)[:,None]
    peak=np.max(np.abs(music))
    if peak:music*=.65/peak
    write_wave(destination,music)
    return {'composition':'New scene-length sampled score using the original studio harmonic palettes',
            'sampler':'TinySoundFont','bank':bank.name,'tempo':bpm,'notes':notes,'seconds':seconds}

def wood_knocks(seconds:float,starts:list[float],level:float=.34)->np.ndarray:
    audio=np.zeros((round(seconds*SR),2),np.float32);rng=np.random.default_rng(217)
    n=round(.19*SR);t=np.arange(n)/SR
    for start in starts:
        sound=(np.sin(2*np.pi*118*t)*np.exp(-28*t)+.4*np.sin(2*np.pi*327*t)*np.exp(-45*t)
               +rng.normal(0,.18,n)*np.exp(-85*t))*level
        pos=round(start*SR);length=min(n,len(audio)-pos)
        if length>0:audio[pos:pos+length]+=sound[:length,None]
    return audio
