"""Hash actual decoded RGB frames; do not confuse codec timestamps with motion."""
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path


def decoded_pixel_evidence(video:Path,width=576,height=1024,frames=33):
    size=width*height*3
    process=subprocess.Popen(['ffmpeg','-v','error','-i',str(video),'-an',
        '-pix_fmt','rgb24','-f','rawvideo','-'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    hashes=[]
    try:
        while True:
            data=process.stdout.read(size)
            if not data:break
            if len(data)!=size:raise ValueError('Partial decoded native RGB frame')
            hashes.append(hashlib.sha256(data).hexdigest())
            if len(hashes)>frames:raise ValueError('More decoded frames than native pilot contract')
        if process.wait(timeout=30) or len(hashes)!=frames:
            raise ValueError('Full native RGB decode failed')
    finally:
        if process.stdout:process.stdout.close()
        if process.poll() is None:
            process.kill();process.wait(timeout=10)
    return {'unique_decoded_rgb_frames':len(set(hashes)),
            'adjacent_exact_duplicate_pairs':sum(a==b for a,b in zip(hashes,hashes[1:])),
            'decoded_frame_hash_chain_sha256':hashlib.sha256(''.join(hashes).encode()).hexdigest(),
            'nonidentical_pixels_do_not_prove_natural_acting':True}
