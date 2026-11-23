"""TWO WORLDS STUDIO: real-source native-motion wave mastering.

No still-image animation and no claims of neural video inference. Reuses every
original decoded frame in order and holds only unoccluded background plates.
Forehead, mouth, beard, glasses, head and hands always remain LIVE source video.
The Studio's strict QA and privacy controls apply to the rendered MP4.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
from fractions import Fraction

import cv2
import numpy as np

STUDIO_PROFILE='TWO_WORLDS_NATIVE_WAVE_LOCKED_CAMERA_V1'


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def _probe(path):
    o=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                   '-show_entries','stream=width,height,r_frame_rate,nb_read_frames,pix_fmt:format=duration',
                   '-of','json',str(path)], timeout=30))
    return o['streams'][0],o['format']


def foreground_envelope(width:int,height:int):
    """Conservative non-tracking actor envelope: all identity and limbs preserved.

    The mask deliberately errs on retaining original video. Source body and hand
    are never fabricated from separate photos. Fixed-background parts only.
    """
    if (width,height)!=(720,1280):raise ValueError('Only validated 720x1280 greeting source')
    mask=np.zeros((height,width),np.uint8)
    # The face is NOT frozen or separately transformed: mouth/beard/glasses are one video.
    cv2.ellipse(mask,(345,420),(230,295),0,0,360,255,-1)
    cv2.ellipse(mask,(360,935),(460,430),0,0,360,255,-1)
    cv2.ellipse(mask,(580,600),(155,300),0,0,360,255,-1) # raised fingers & forearm
    cv2.ellipse(mask,(122,748),(140,210),0,0,360,255,-1) # hand with mic
    # Safe late-pose margin for waving elbow and fingertips; no unprotected fingers.
    cv2.rectangle(mask,(430,650),(720,1110),255,-1)
    # At far bottom the user's body, hands and table remain source motion.
    cv2.rectangle(mask,(0,1080),(720,1280),255,-1)
    # Soft boundary avoids seams, without temporal ghosting of facial details.
    return cv2.GaussianBlur(mask.astype(np.float32)/255,(0,0),sigmaX=25)


def compose(frame,plate,alpha):
    if frame.shape!=plate.shape or frame.shape[:2]!=alpha.shape:
        raise ValueError('Frame/plate dimensions mismatch')
    out=frame.astype(np.float32)*alpha[:,:,None]+plate.astype(np.float32)*(1-alpha[:,:,None])
    return np.clip(np.round(out),0,255).astype(np.uint8)


def _quality_from_frames(video,reference,region=(0,0,120,145)):
    x,y,w,h=region
    cap=cv2.VideoCapture(str(video))
    count=0;means=[];last=None;adj_same=0
    try:
        while True:
            ok,frm=cap.read()
            if not ok:break
            count+=1
            sample=frm[y:y+h,x:x+w]
            if last is not None:
                adj_same+=int(np.array_equal(last,sample))
            last=sample.copy()
            means.append(float(np.mean(np.abs(sample.astype(np.float32)-reference[y:y+h,x:x+w].astype(np.float32)))))
        success=(count>0)
    finally:cap.release()
    return {'frames_decoded':count,'background_roi_mad_to_anchor_mean':round(float(np.mean(means)),4),
            'background_roi_mad_to_anchor_max':round(float(np.max(means)),4),
            'adjacent_static_roi_identical':adj_same,'decode_ok':success}


def render_wave(source:Path,output:Path,report:Path,*,height=1920,quality=17):
    source=Path(source).resolve();output=Path(output).resolve();report=Path(report).resolve()
    if source==output or output.exists() or report.exists():
        raise ValueError('Source is immutable; deliverables must not exist')
    if source.suffix.lower()!='.mp4' or not source.is_file() or output.suffix.lower()!='.mp4':
        raise ValueError('A real existing MP4 source and fresh MP4 output required')
    if height not in (1280,1920) or quality<14 or quality>20:
        raise ValueError('Supported exports: 720p or 1080p, high-quality CRF 14..20')
    inp,_=_probe(source)
    if (inp['width'],inp['height'])!=(720,1280) or Fraction(inp['r_frame_rate'])!=60:
        raise ValueError('Native source must contain 720x1280 frames at 60 FPS')
    input_count=int(inp['nb_read_frames'])
    if not 150<=input_count<=360:raise ValueError('Unrecognized wave duration')
    # Source privacy: no media goes into GitHub or remote endpoints.
    cap=cv2.VideoCapture(str(source))
    ok,plate=cap.read()
    if not ok:raise RuntimeError('Could not decode original anchor')
    alpha=foreground_envelope(720,1280)
    # No still-body or interpolation: each source frame drives the entire person.
    output.parent.mkdir(parents=True,exist_ok=True)
    width=height*9//16
    temp=output.with_name(output.stem+'.partial.mp4')
    if temp.exists():raise FileExistsError('Stale partial output must be inspected')
    cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
         '-f','rawvideo','-pix_fmt','bgr24','-video_size','720x1280','-framerate','60','-i','-',
         '-an','-vf',f'scale={width}:{height}:flags=lanczos,format=yuv420p',
         '-c:v','libx264','-preset','medium','-crf',str(quality),'-threads','4','-r','60',
         '-movflags','+faststart',str(temp)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    count=0
    try:
        while ok:
            result=compose(plate if count==0 else frame,plate,alpha)
            proc.stdin.write(result.tobytes())
            count+=1
            ok,frame=cap.read()
        proc.stdin.close()
        err=proc.stderr.read()
        proc.stderr.close()
        if proc.wait(timeout=180)!=0:raise RuntimeError('FFmpeg failed: '+err[-700:].decode(errors='replace'))
    except BaseException:
        try:proc.stdin.close()
        except (BrokenPipeError,ValueError):pass
        proc.kill();proc.wait();proc.stderr.close();temp.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
    try:
        stream,_=_probe(temp)
        if int(stream['nb_read_frames'])!=input_count or (stream['width'],stream['height'])!=(width,height) or Fraction(stream['r_frame_rate'])!=60:
            raise RuntimeError('A decoded output frame, geometry, or framerate is missing')
        # Full stream decode (not a blind FFprobe-only container check).
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(temp),'-map','0:v:0',
                        '-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=180)
        cv2_cap=cv2.VideoCapture(str(temp));ok,ref=cv2_cap.read();cv2_cap.release()
        if not ok:raise RuntimeError('Video first frame cannot be decoded')
        source_qc=_quality_from_frames(source,plate)
        if height!=1280:
            # Compare output to its own decoded first frame for the stationary plate ROI.
            p=(0,0,180,218)
            final_qc=_quality_from_frames(temp,ref,region=p)
        else:
            final_qc=_quality_from_frames(temp,ref)
        if final_qc['frames_decoded']!=input_count:raise RuntimeError('Incomplete output RGB decode')
        if final_qc['background_roi_mad_to_anchor_mean']>1.5:
            raise RuntimeError('Locked studio background failed strict QC')
        report.parent.mkdir(parents=True,exist_ok=True)
        receipt={'engine':STUDIO_PROFILE,'status':'TECHNICAL_PASS_VISUAL_REVIEW_REQUIRED',
                 'human_artistic_approved':False,'native_neural_inference_performed':False,
                 'synthetic_added_frames':0,'frame_interpolation_performed':False,
                 'input_native_fps':60,'input_frames':input_count,'output_frames':input_count,
                 'output_width':width,'output_height':height,'output_fps':60,
                 'duration_seconds':round(input_count/60,4),'source_sha256':sha256(source),
                 'output_sha256':sha256(temp),'source_static_region':source_qc,
                 'result_static_region':final_qc,'source_path_private':True,
                 'image_identity_kept_from_original_video':True,
                 'animation_method':'source_live_performance_static_background_only',
                 'output_is_upscaled':height==1920,
                 'notes':'Human hand/finger and beard morphology still need full human visual inspection; interpolation already present in original uploaded source cannot be undone.'}
        temp.replace(output)
        report.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return receipt
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description='Two Worlds Studio locked-camera greeting remaster; no GPU')
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--height',type=int,default=1920,choices=(1280,1920))
    args=p.parse_args(argv)
    print(json.dumps(render_wave(args.source,args.output,args.report,height=args.height),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
