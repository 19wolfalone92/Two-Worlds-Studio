"""Recorded voice → deterministic, minimal 2D mouth cue envelope.

Not phoneme recognition. The art team supplies an approved transparent
open-mouth overlay registered to the actor sprite. We only calculate on/off
timing from RMS energy. This is a subtle low-budget motion-comic technique.
"""
from __future__ import annotations
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

def voice_envelope(path:Path, step_ms=50)->np.ndarray:
    if not (20<=step_ms<=100):
        raise ValueError("Envelope step must be 20..100ms")
    output=subprocess.run(["ffmpeg","-nostdin","-v","error",
            "-i",str(path),"-vn","-ac","1","-ar","16000",
            "-f","s16le","-"],capture_output=True,check=True).stdout
    if len(output)>16000*2*600:
        raise ValueError("Voice sample exceeds safety maximum")
    if len(output)<640:
        raise ValueError("Voice sample too short")
    signal=np.frombuffer(output[:len(output)//2*2],dtype=np.int16).astype(np.float32)/32768
    chunk=round(16000*step_ms/1000)
    level=np.asarray([float(np.sqrt(np.mean(signal[i:i+chunk]**2)))
                      for i in range(0,len(signal),chunk)],dtype=np.float32)
    # First-order smoothing removes distracting mechanical mouth chattering.
    if len(level)>2:
        level=np.convolve(level,[.2,.6,.2],mode="same").astype(np.float32)
    peak=float(np.percentile(level,95)) if len(level) else 0.0
    if peak<=1e-5:
        return np.zeros_like(level)
    return np.clip(level/peak,0,1)

def mouth_open(envelope:np.ndarray,scene_seconds:float,*,step_ms=50,
               threshold=.22,delay_ms=0)->bool:
    if not len(envelope):
        return False
    if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not .05<=threshold<=.8:
        raise ValueError("Mouth threshold outside usable range")
    index=math.floor((scene_seconds*1000-delay_ms)/step_ms)
    if index<0 or index>=len(envelope):
        return False
    return bool(envelope[index]>=threshold)

def main():
    p=argparse.ArgumentParser(description="Inspect sound-driven mouth motion cues")
    p.add_argument("--voice",type=Path,required=True)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    values=voice_envelope(args.voice).tolist()
    data={"voice":str(args.voice),"step_ms":50,"levels":values}
    content=json.dumps(data,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(content,encoding="utf-8")
    print(content)

if __name__=="__main__":
    main()
