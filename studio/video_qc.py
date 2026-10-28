"""CPU-friendly automatic video/audio diagnostics, NOT professional aesthetic proof."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

def sample_video(video:Path, *, fps:float=2.0, sample_width:int=72,
                 sample_height:int=128,max_frames:int=1200)->dict:
    if not 0<fps<=8:
        raise ValueError("Diagnostic sample FPS must be 0..8")
    args=["ffmpeg","-nostdin","-v","error","-i",str(video),"-vf",
          f"fps={fps},scale={sample_width}:{sample_height},format=gray",
          "-f","rawvideo","-pix_fmt","gray","-"]
    run=subprocess.run(args,check=True,capture_output=True)
    size=sample_width*sample_height
    if len(run.stdout)%size:
        raise RuntimeError("Diagnostic output frame is truncated")
    n=len(run.stdout)//size
    if n==0:
        raise ValueError("No diagnostic frames")
    if n>max_frames:
        raise ValueError("Too many frames: shorten clip or lower diagnostic FPS")
    data=np.frombuffer(run.stdout,dtype=np.uint8).reshape((n,sample_height,sample_width))
    avg=np.mean(data,axis=(1,2))
    motion=(np.abs(np.diff(data.astype(np.int16),axis=0)).mean(axis=(1,2))
             if n>1 else np.array([],dtype=np.float64))
    dark=int(np.count_nonzero(avg<12))
    white=int(np.count_nonzero(avg>247))
    still=int(np.count_nonzero(motion<1.0))
    notes=[]
    if dark/n>.24:
        notes.append("More than 24% nearly black sampled frames; review intentional fades")
    if white/n>.16:
        notes.append("More than 16% nearly white frames; possible blown-out flashes")
    if n>2 and still/max(1,n-1)>.86:
        notes.append("Little interframe motion; verify actual animated presentation")
    if np.percentile(avg,95)-np.percentile(avg,5)<8:
        notes.append("Very narrow scene brightness range, check visual contrast")
    return {"frames_sampled":n,"nearly_black":dark,"nearly_white":white,
            "still_adjacent_samples":still,
            "mean_brightness":round(float(avg.mean()),2),
            "mean_motion":round(float(motion.mean()),3) if n>1 else 0,
            "notes":notes,"automated_aesthetic_approval":False}

def report(video:Path)->dict:
    data=sample_video(video)
    data["video"]=video.name
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video",required=True,type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    d=report(args.video)
    out=json.dumps(d,ensure_ascii=False,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(out,encoding="utf-8")
    print(out)

if __name__=="__main__":
    main()
