#!/usr/bin/env python3
"""Private Work v1.1 art -> original TWO WORLDS 10s acting/mirror benchmark.

No story edits, no publication. Prepares two cutout elbow/shoulder transforms
from the PRIVATE existing Work PNGs, never commits art to public GitHub.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess, wave, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2

W,H=1080,1920
ASSET_PREFIX='TWO_WORLDS_Studio_Handoff_v1_1/assets/E01/shared/'
VOICE_PREFIX='TWO_WORLDS_Studio_Handoff_v1_1/assets/E01/voices/'
REQUIRED={'earth_blackout.png','mirror_noctaris_room_rgba.png','lyra_inside_mirror_rgba.png','mira_dialogue_rgba.png','mirror_frame_rgba.png','mira_contact_hand_rgba.png','lyra_contact_hand_rgba.png'}

def art(zip_path,scale=1):
    with zipfile.ZipFile(zip_path) as z:
        names=set(z.namelist())
        if any(ASSET_PREFIX+x not in names for x in REQUIRED):
            raise ValueError('Missing approved Work reference assets; do not fake them')
        images={}
        for name in REQUIRED:
            import io
            with Image.open(io.BytesIO(z.read(ASSET_PREFIX+name))) as im:
                im.load()
                if im.size!=(W,H): raise ValueError('Cannot change Work art dimensions')
                if scale!=1:im=im.resize((round(W*scale),round(H*scale)),Image.Resampling.LANCZOS)
                images[name]=np.array(im.convert('RGBA'),dtype=np.uint8)
        audio=z.read(VOICE_PREFIX+'S03_Mira_af_heart.wav')
        return images,audio

def prep_arm(actor:np.ndarray,which:str):
    h,w=actor.shape[:2]
    # Polygon lasso based on exact known Work v1.1 cutouts, NOT a generic
    # "AI smart arm" segmentation that could silently break another asset.
    if which=='mira':
        poly=[(0,746),(225,744),(256,828),(283,904),(306,966),
              (261,1004),(261,1051),(240,1213),(208,1326),(0,1328)]
        pivot=(256,940)
    else:
        poly=[(656,553),(900,487),(971,817),(864,900),(726,833),
              (688,743),(630,656)]
        pivot=(671,641)
    f=w/W
    mask=np.zeros((h,w),np.uint8)
    cv2.fillPoly(mask,[np.asarray([(round(x*f),round(y*f)) for x,y in poly],dtype=np.int32)],255)
    # Preserve seamless art on the shoulder seam with a gentle 2-pixel alpha transition.
    mask=cv2.GaussianBlur(mask,(5,5),1.25)
    arm=actor.copy();arm[:,:,3]=(arm[:,:,3].astype(np.uint16)*mask//255).astype('uint8')
    torso=actor.copy();torso[:,:,3]=(torso[:,:,3].astype(np.uint16)*(255-mask)//255).astype('uint8')
    return torso,arm,(pivot[0]*f,pivot[1]*f)

def linear(a,b,t):return a+(b-a)*t

def eased(t):
    t=max(0,min(1,t));return t*t*(3-2*t)

def roto(im:np.ndarray,center,angle,scale=1,dx=0,dy=0):
    h,w=im.shape[:2]
    m=cv2.getRotationMatrix2D(center,angle,scale)
    m[0,2]+=dx;m[1,2]+=dy
    return cv2.warpAffine(im,m,(w,h),flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))

def composite(base:np.ndarray,fg:np.ndarray)->np.ndarray:
    # Uint16 alpha blending is stable and avoids dark fringe from un-premultiplied alpha.
    a=fg[:,:,3:4].astype(np.float32)/255.0
    base=base.astype(np.float32)
    fg=fg.astype(np.float32)
    out=base[:,:,:3]*(1-a)+fg[:,:,:3]*a
    return np.uint8(np.clip(out,0,255))

def alpha_over(bg:np.ndarray,fg:np.ndarray)->np.ndarray:
    b=bg.astype(np.float32)/255
    f=fg.astype(np.float32)/255
    aa=f[:,:,3:4]+b[:,:,3:4]*(1-f[:,:,3:4]);safe=np.maximum(aa,1e-7)
    col=(f[:,:,:3]*f[:,:,3:4]+b[:,:,:3]*b[:,:,3:4]*(1-f[:,:,3:4]))/safe
    return np.uint8(np.clip(np.concatenate((col,aa),axis=2)*255,0,255))

class Scene:
    def __init__(self,images):
        self.h,self.w=next(iter(images.values())).shape[:2]
        self.f=self.w/W
        self.room=images['earth_blackout.png']
        self.inside=images['mirror_noctaris_room_rgba.png']
        self.frame=images['mirror_frame_rgba.png']
        self.mira_hand=images['mira_contact_hand_rgba.png']
        self.lyra_hand=images['lyra_contact_hand_rgba.png']
        self.lyra_body,self.lyra_arm,self.lyra_pivot=prep_arm(images['lyra_inside_mirror_rgba.png'],'lyra')
        self.mira_body,self.mira_arm,self.mira_pivot=prep_arm(images['mira_dialogue_rgba.png'],'mira')
        yy,xx=np.mgrid[0:self.h,0:self.w]
        # Only the mirror oval receives animated glass reflections.
        cx,cy=548*self.f,875*self.f
        self.inside_oval=np.maximum(0,1-(((xx-cx)/(388*self.f))**2+((yy-cy)/(720*self.f))**2))
        self.glass=np.float32(np.clip(self.inside_oval*1.2,0,1))
        self.rain_phase=(xx*.007+yy*.0035)
        self.vignette=(1-.2*np.minimum(1,((xx-self.w/2)/(self.w/2))**2+((yy-self.h/2)/(self.h/2))**2))[:,:,None]
        self.shadow_oval=np.uint8(np.clip(np.exp(-(((xx-750*self.f)/(255*self.f))**2+((yy-1700*self.f)/(65*self.f))**2))*65,0,255))
        self.dark_mira=np.array([.81,.84,.89,1.0],dtype=np.float32)
        self.dark_lyra=np.array([.89,.9,.98,1.0],dtype=np.float32)
    def mirror_actor_frame(self,sec):
        f=self.f; t=max(0,min(sec/10,1))
        entry=eased((sec-.15)/4.8)
        armraise=eased((sec-2.4)/4.2)
        lyra_move=eased((sec-1.7)/4.3)
        # Original Work art and its internal spatial order, no horizontal mirroring.
        room=self.room.copy()
        if sec<4.9:
            phase=.00035*sec
            room=cv2.warpAffine(room,np.float32([[1,0,phase*self.w],[0,1,0]]),(self.w,self.h),borderMode=cv2.BORDER_REPLICATE)
        room=alpha_over(room,self.inside)
        # Noctaris actor has real articulated right-arm movement at original shoulder.
        lyra=roto(self.lyra_arm,self.lyra_pivot,linear(0,10,lyra_move))
        lyra=alpha_over(self.lyra_body,lyra)
        # Subtle true actor translation and leg stance as opposed to background zoom alone.
        lyra=roto(lyra,(546*f,1080*f),.6*math.sin(sec*.65),1.0,
                   dx=3.2*f*math.sin(sec*.42),dy=1.6*f*math.sin(sec*.65))
        lyra[:,:,0:3]=(lyra[:,:,0:3].astype(np.float32)*self.dark_lyra[:3]).astype('uint8')
        # Cast subtle room shadow from actor — torso clips into Noctaris interior.
        room=alpha_over(room,lyra)
        # Animate glass surface reflections in FRONT of Lyra but behind the frame.
        scan=self.glass*(.015+.018*np.sin(sec*1.3))
        if scan.max():
            rgb=room[:,:,:3].astype(np.float32)
            rgb[:,:,0]=np.clip(rgb[:,:,0]+scan*78,0,255)
            rgb[:,:,1]=np.clip(rgb[:,:,1]+scan*99,0,255)
            rgb[:,:,2]=np.clip(rgb[:,:,2]+scan*161,0,255)
            room[:,:,:3]=rgb.astype('uint8')
        room=alpha_over(room,self.frame)
        # Physical Mira walks into foreground. Body and raised arm move on independent
        # keyed joint. All art comes from canonical Work original, not replacement faces.
        mira_arm=roto(self.mira_arm,self.mira_pivot,linear(-28,7,armraise))
        mira=alpha_over(self.mira_body,mira_arm)
        # Actor advances from the dark bedroom toward mirror. Foreground gets larger,
        # directional shadow falls on the room side, and foreground light dims.
        scale=linear(.68,.81,entry)
        dx=linear(435,355,entry)*f
        dy=linear(312,260,entry)*f
        mira=roto(mira,(0,0),0,scale,dx,dy)
        mira[:,:,0:3]=np.uint8(np.clip(mira[:,:,:3].astype(np.float32)*self.dark_mira[:3],0,255))
        # Shadow is under the foreground character; visible on left floor only.
        frame=room[:,:,:3].copy()
        shadow=self.shadow_oval.astype(np.float32)/255
        frame=np.uint8(np.clip(frame.astype(np.float32)*(1-shadow[:,:,None]*.52),0,255))
        frame=composite(frame,mira)
        # Very subtle frame-coherent lighting, no random pixel glitter / eye deformation.
        bloom=6*math.exp(-((sec-8.75)/.75)**2)
        if bloom:
            gl=self.glass*bloom
            frame=np.uint8(np.clip(frame.astype(np.float32)+gl[:,:,None]*np.array([.56,.70,.95]),0,255))
        frame=np.uint8(np.clip(frame.astype(np.float32)*self.vignette,0,255))
        # The real Work sprite ends before the bottom of the vertical canvas.
        # Push the camera into the two faces so that clipping is naturally
        # outside frame rather than creating floating sticker-cutout legs.
        focus=(.52*self.w,.38*self.h)
        mat=cv2.getRotationMatrix2D(focus,0,1.28)
        return cv2.warpAffine(frame,mat,(self.w,self.h),
                             flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    def palms_frame(self,sec):
        # Exact Work S08 photo-cutout palms and their original left/right
        # physical bodies. The glass is represented as a fixed plane: hands
        # approach but never intersect or pass through the glass.
        f=self.f
        t=eased((sec-5.8)/3.0)
        blue=self.mira_hand
        violet=self.lyra_hand
        shift=linear(155,0,t)*f
        # Palm translation plus small wrist pivots are two actual actor motions.
        left=roto(blue,(270*f,1270*f),linear(4,-1,t),dx=-shift,
                  dy=linear(33,0,t)*f)
        right=roto(violet,(805*f,1260*f),linear(-4,1,t),dx=shift,
                   dy=linear(33,0,t)*f)
        bg=self.room[:,:,:3].copy().astype(np.float32)
        # Physically distinct light from Earth to the left, Noctaris on right.
        half=np.arange(self.w,dtype=np.float32)[None,:,None]
        grad=np.clip((half-self.w*.47)/(self.w*.12),0,1)
        bg=bg*(1-grad*.22)+grad*np.array([12,12,21],dtype=np.float32)
        bg=np.uint8(np.clip(bg,0,255))
        bg=composite(bg,left)
        bg=composite(bg,right)
        # Straight reflective glass between hands. Do not invent transport.
        height=np.arange(self.h,dtype=np.float32)[:,None]
        center=np.arange(self.w,dtype=np.float32)[None,:]
        plane=np.exp(-((center-self.w*.5)/(7*f))**2)*np.clip((height-230*f)/(120*f),0,1)*np.clip((1520*f-height)/(120*f),0,1)
        intensity=(.09+.26*(t**2))*plane
        bg=np.uint8(np.clip(bg.astype(np.float32)+intensity[:,:,None]*np.array([53,85,140]),0,255))
        # Only becomes bright when both hands have almost reached the glass.
        if t>.75:
            intensity2=(t-.75)*14*np.exp(-((center-self.w*.5)/(145*f))**2)*np.exp(-((height-530*f)/(300*f))**2)
            bg=np.uint8(np.clip(bg.astype(np.float32)+intensity2[:,:,None]*np.array([.48,.62,.85]),0,255))
        return np.uint8(np.clip(bg.astype(np.float32)*self.vignette,0,255))
    def frame_at(self,sec):
        if sec<5.65:
            return self.mirror_actor_frame(sec)
        if sec>6.2:
            return self.palms_frame(sec)
        d=eased((sec-5.65)/.55)
        a=self.mirror_actor_frame(sec)
        b=self.palms_frame(sec)
        return cv2.addWeighted(a,1-d,b,d,0)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--handoff',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fps',type=int,default=24)
    p.add_argument('--scale',type=float,default=1.)
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--frames',type=Path,default=None)
    a=p.parse_args()
    images,voice=art(a.handoff,a.scale)
    scene=Scene(images)
    if a.frames:
        a.frames.mkdir(parents=True,exist_ok=True)
        for sec in (0,2,4,6,8,9.5):
            im=scene.frame_at(sec)
            Image.fromarray(im,'RGB').save(a.frames/f'mirror_{sec:04.1f}.jpg',quality=91)
    w,h=scene.w,scene.h
    a.output.parent.mkdir(parents=True,exist_ok=True)
    duration=a.seconds
    # Soundtrack from original pilot voice only (do not create misleading new dialogue).
    wav=a.output.with_name('mira_original_voice.wav')
    wav.write_bytes(voice)
    ff=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(a.fps),'-i','-',
        '-itsoffset','2.15','-i',str(wav),
        '-filter_complex',f'[0:v]scale=1080:1920:flags=lanczos,format=yuv420p[v];'
             f'anoisesrc=color=brown:amplitude=0.008:sample_rate=48000:d={duration}[noise];'
             '[noise]lowpass=f=130,volume=0.28[amb];[1:a]aresample=48000,volume=1.45[voc];'
             f'[amb][voc]amix=inputs=2:duration=longest:dropout_transition=0[a]',
        '-map','[v]','-map','[a]','-t',str(duration),'-c:v','libx264','-preset','veryfast',
        '-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k',
        '-movflags','+faststart',str(a.output)]
    proc=subprocess.Popen(ff,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    n=round(a.fps*duration)
    try:
        for i in range(n):
            im=scene.frame_at(i/a.fps)
            proc.stdin.write(im.tobytes())
        proc.stdin.close()
        stderr=proc.stderr.read().decode('utf-8',errors='replace')
        code=proc.wait(timeout=80)
        if code:raise RuntimeError('FFmpeg exit %d: %s'%(code,stderr[-2000:]))
    except BaseException:
        proc.kill();a.output.unlink(missing_ok=True);raise
    wav.unlink(missing_ok=True)
    stats={'status':'OWNER_REVIEW_ONLY','duration_seconds':duration,'frames':n,
           'resolution':'1080x1920','actual_art':'Work v1.1 original Mira and Lyra',
           'animation':'Two separately animated actor arm/shoulder joints, root motion, depth-aware mirror foreground, light and shadow',
           'limitations':'Manual 2D joint separation and simple pose changes; not true walking physics or automatic body rigging, not full animation',
           'published':False,'qa_source_sha256':hashlib.sha256(a.handoff.read_bytes()).hexdigest(),
           'mp4_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}
    a.output.with_suffix('.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
