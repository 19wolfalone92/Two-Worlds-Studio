"""Original procedural environmental sound beds, never claimed as field recordings."""
from __future__ import annotations
import hashlib
import json
import wave
from pathlib import Path
import numpy as np

KINDS={'quiet_room','domestic_room','rain_coach','habitat_fan','greenhouse','clock_shop'}


def create_ambience(dest: Path,kind: str,*,seconds=20.0,sample_rate=44100,seed=0) -> dict:
    if kind not in KINDS or not 5<=seconds<=120 or dest.exists():
        raise ValueError('Unknown ambience, invalid duration or existing destination')
    rng=np.random.default_rng(seed)
    n=round(seconds*sample_rate)
    t=np.arange(n,dtype=np.float64)/sample_rate
    frequencies=np.fft.rfftfreq(n,1/sample_rate)
    def noise(power,low,high):
        spectrum=rng.normal(size=len(frequencies))+1j*rng.normal(size=len(frequencies))
        weight=np.maximum(frequencies,10)**(-power/2)
        weight*=1/(1+(low/np.maximum(frequencies,1))**6)
        weight*=1/(1+(frequencies/high)**6)
        spectrum*=weight;spectrum[0]=0
        signal=np.fft.irfft(spectrum,n=n).astype(np.float32)
        return signal/max(float(np.sqrt(np.mean(signal**2))),1e-9)
    left=noise(.9,30,4500)
    right=.82*left+.18*noise(.9,30,4500)
    level=.004
    signal=np.stack([left,right],axis=1)*level
    if kind=='rain_coach':
        rain=noise(.55,500,8500)*(.006+.002*np.sin(2*np.pi*t*(2/seconds)))
        engine=sum(np.sin(2*np.pi*f*t)*a for f,a in [(43,.004),(86,.0015),(129,.0008)])
        signal+=np.stack([rain+engine,.92*rain+engine],axis=1)
        # Small damped water impacts; no sampled or synthesized human voices.
        for when in rng.uniform(.2,seconds-.4,90):
            length=round(rng.uniform(.018,.08)*sample_rate)
            u=np.arange(length)/sample_rate
            drop=rng.normal(size=length)*np.exp(-u*90)*.009
            pos=round(when*sample_rate)
            pan=rng.uniform(.15,.85)
            signal[pos:pos+length,0]+=drop*(1-pan)
            signal[pos:pos+length,1]+=drop*pan
    elif kind=='habitat_fan':
        fan=noise(1.1,90,1800)*.004
        hum=(np.sin(2*np.pi*100*t)+.3*np.sin(2*np.pi*200*t))*.0015
        signal+=np.stack([fan+hum,.9*fan+hum],axis=1)
    elif kind=='domestic_room':
        hum=np.sin(2*np.pi*50*t)*.00065
        signal+=hum[:,None]
    elif kind=='greenhouse':
        wind=noise(1.7,25,1700)*(.004+.0015*np.sin(2*np.pi*t/seconds))
        signal+=np.stack([wind,.85*wind],axis=1)
    elif kind=='clock_shop':
        for index,when in enumerate(np.arange(.25,seconds-.15,.5)):
            count=round(.08*sample_rate);u=np.arange(count)/sample_rate
            click=(rng.normal(size=count)*np.exp(-u*180)
                   +np.sin(2*np.pi*(1850 if index%2 else 1450)*u)*np.exp(-u*100))*.012
            pos=round(when*sample_rate)
            signal[pos:pos+count,0]+=click
            signal[pos:pos+count,1]+=click*.85
    signal-=signal.mean(axis=0)
    fade=np.minimum(1,np.minimum(t/.12,(seconds-t)/.12)).clip(0,1)
    signal*=fade[:,None]
    dest.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(dest),'wb') as wav:
        wav.setnchannels(2);wav.setsampwidth(2);wav.setframerate(sample_rate)
        wav.writeframes((signal.clip(-1,1)*32767).astype('<i2').tobytes())
    with dest.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    receipt={'kind':kind,'source':'original_procedural_sound_design','sampled_recordings':False,
             'seconds':n/sample_rate,'sample_rate':sample_rate,'seed':seed,'sha256':digest,
             'peak':round(float(np.abs(signal).max()),6),'rms':round(float(np.sqrt(np.mean(signal**2))),6),
             'listening_review':'pending','published':False}
    dest.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2))
    return receipt
