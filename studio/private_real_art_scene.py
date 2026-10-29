"""PRIVATE real-art mirror motion benchmark for TWO WORLDS Studio.

Uses a private, owner-supplied single composite with Mira's BACK visible and
Lyra beyond mirror glass. Animated body deformation is continuous and modest.
This is an ACTUAL 10s 24fps MP4 render, *not* full character rigging/acting.
No art, originals or voice clips are committed into the public repository.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


def ease(t):
    t = max(0., min(1., float(t)))
    return t*t*(3-2*t)


class MirrorMotionScene:
    """Deterministic pixel-field motions specific to existing mirror composition."""
    def __init__(self, source: Image.Image, width: int = 720, height: int = 1280):
        if cv2 is None:
            raise RuntimeError("Install requirements-cinematic.txt")
        if width < 160 or height < 284 or width*height > 2_200_000:
            raise ValueError("Working frame size unsafe")
        if abs(source.width/source.height-941/1672) > .025:
            raise ValueError("Expected the vertical TWO WORLDS original mirror staging")
        self.w, self.h = width, height
        src = np.array(source.convert("RGB"), dtype=np.uint8)
        self.scene = cv2.resize(src, (width, height), interpolation=cv2.INTER_AREA)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        self.xx, self.yy = xx, yy
        self.x = xx/width
        self.y = yy/height
        def g(cx,cy,rx,ry):
            return np.exp(-.5*(((self.x-cx)/rx)**2+((self.y-cy)/ry)**2)).astype(np.float32)
        self.mira_upper = g(.295,.40,.205,.19)
        self.mira_head = g(.335,.245,.145,.09)
        self.mira_shoulder = g(.35,.40,.17,.12)
        self.mira_arm = g(.588,.68,.075,.16)
        self.mira_hip = g(.34,.71,.21,.16)
        self.lyra_head = g(.70,.28,.097,.105)
        self.lyra_shoulder = g(.74,.48,.12,.17)
        self.mirror = np.clip(1-(((self.x-.71)/.25)**2+
                                  ((self.y-.34)/.34)**2),0,1)**2
        self.vignette = 1-.13*np.clip(((self.x-.5)/.60)**2+
                                       ((self.y-.52)/.75)**2,0,1)
        self.rim = self.mira_shoulder*.64+self.mira_head*.37
        self.lantern = g(.81,.29,.072,.13)
        self.drops = np.random.default_rng(2118).random((80,3))

    def frame(self, seconds: float) -> np.ndarray:
        if not math.isfinite(seconds) or not 0 <= seconds <= 10:
            raise ValueError("Mirror seconds must be finite in 0..10")
        reach=ease((seconds-4.7)/3.2)
        approach=ease((seconds-1)/4.2)
        breath=math.sin(seconds*2.15)
        dx = (self.mira_upper*(1.5*breath+6*approach)
              +self.mira_head*(2.3*math.sin(seconds*.58)+3.1*approach)
              +self.mira_shoulder*3*math.sin(seconds*.75)
              +self.mira_arm*46*reach
              +self.mira_hip*1.1*math.sin(seconds*1.3)
              +self.lyra_head*1.8*math.sin(seconds*.8)
              -self.lyra_shoulder*3*ease((seconds-3)/2.6))
        dy = (self.mira_upper*(-2*approach-1.5*breath)
              +self.mira_head*(-1.5*approach+1.5*math.sin(seconds*.52))
              -self.mira_arm*72*reach
              +self.lyra_head*1.9*ease((seconds-3.1)/2.4))
        # Deforms original PHOTO-LIKE pixel field (no floating sprites);
        # not motion capture, neural frame synthesis or genuine walking IK.
        frame=cv2.remap(self.scene,(self.xx-dx).astype(np.float32),
                        (self.yy-dy).astype(np.float32),cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
        phase=.5+.5*math.sin(seconds*2.4)
        glass=self.mirror*(1.3+3.2*phase+5.5*math.exp(-((seconds-8.4)/.8)**2))
        frame[:,:,0]+=glass*.36
        frame[:,:,1]+=glass*.64
        frame[:,:,2]+=glass*1.4
        rim=self.rim*(1+reach)*1.3
        frame[:,:,0]+=rim*.6
        frame[:,:,1]+=rim*.8
        frame[:,:,2]+=rim*1.5
        lantern=self.lantern*(1+.3*math.sin(2.2*seconds))*2.2
        frame[:,:,0]+=lantern
        frame[:,:,1]+=lantern*.22
        frame*=self.vignette[:,:,None]
        if seconds>=6:
            zoom=1.08+.04*ease((seconds-6)/4)
            center=(.575*self.w,.385*self.h)
        else:
            zoom=1.018+.062*ease(seconds/6)
            center=(.515*self.w,.41*self.h)
        mat=cv2.getRotationMatrix2D(center,0,zoom)
        frame=cv2.warpAffine(np.uint8(np.clip(frame,0,255)),mat,
                            (self.w,self.h),flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
        # Rain moves only in the Earth's window; no fake extra characters.
        overlay=frame.astype(np.float32)
        for rand_x,start,speed in self.drops:
            x=int((.025+rand_x*.16)*self.w)
            y=int(((start+seconds*(.04+speed*.14))%1)*.47*self.h)
            if y<.03*self.h:
                continue
            cv2.line(overlay,(x,y),(x-1,y+7),(125,170,222),1,cv2.LINE_AA)
        return np.uint8(np.clip(overlay,0,255))


def render_private(source_path:Path, out:Path, *,fps:int=24,
                   seconds:int=10, width:int=720,height:int=1280)->dict:
    if fps !=24 or seconds!=10:
        raise ValueError("This owner-review benchmark is exactly 10s at 24fps")
    if out.exists():
        raise FileExistsError("Owner-review output must not overwrite other videos")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with Image.open(source_path) as image:
        image.load()
        scene=MirrorMotionScene(image,width,height)
    out.parent.mkdir(parents=True,exist_ok=True)
    ff = ['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
          '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
          '-r',str(fps),'-i','-',
          '-f','lavfi','-i','anoisesrc=color=brown:amplitude=0.012:sample_rate=48000',
          '-f','lavfi','-i','sine=frequency=94:sample_rate=48000',
          '-filter_complex','[1:a]lowpass=f=900,volume=0.55[rain];'
                            '[2:a]volume=0.025[hum];'
                            '[rain][hum]amix=inputs=2:duration=first[a]',
          '-map','0:v','-map','[a]','-vf','scale=1080:1920:flags=lanczos,format=yuv420p',
          '-t',str(seconds),'-c:v','libx264','-crf','20','-preset','veryfast',
          '-c:a','aac','-b:a','160k','-movflags','+faststart',str(out)]
    proc=subprocess.Popen(ff,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for i in range(fps*seconds):
            proc.stdin.write(scene.frame(i/fps).tobytes())
        proc.stdin.close()
        stderr=proc.stderr.read().decode('utf-8','replace')
        if proc.wait(timeout=120):
            raise RuntimeError('FFmpeg encode failed: '+stderr[-2500:])
    except BaseException:
        proc.kill()
        out.unlink(missing_ok=True)
        raise
    decoder=subprocess.run(['ffmpeg','-v','error','-i',str(out),'-f','null','-'],
                           capture_output=True,text=True,timeout=90)
    if decoder.returncode:
        out.unlink(missing_ok=True)
        raise RuntimeError('Video full decode failed: '+decoder.stderr[-3000:])
    manifest={
        'status':'REVIEW_ONLY_REAL_ART_PUPPET_ALTERNATIVE',
        'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'source_pixels':'original owner artwork, native resolution disclosed separately',
        'working_resolution':[width,height],
        'export_resolution':[1080,1920], 'upscaled':(width<1080 or height<1920),
        'frames':fps*seconds,'fps':fps,'duration_seconds':seconds,
        'full_mp4_decode':True,'real_character_art':True,
        'fully_articulated_walking':False,'complete_motion_generation':False,
        'professional_quality_approved':False,'owner_approved':False,
        'tiktok_published':False,
        'limitations':'Continuous 2D warp of a single owner reference; no independent joints, no newly inferred back or real motion physics. Human artistic review required.'
    }
    out.with_suffix('.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest


def main():
    parser=argparse.ArgumentParser(description='Private Mira-back-to-camera mirror animation review')
    parser.add_argument('--scene',type=Path,required=True,
                        help='Approved-context ORIGINAL image, Mira seen from behind')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--work-width',type=int,default=720)
    parser.add_argument('--work-height',type=int,default=1280)
    args=parser.parse_args()
    print(json.dumps(render_private(args.scene,args.output,
          width=args.work_width,height=args.work_height),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
