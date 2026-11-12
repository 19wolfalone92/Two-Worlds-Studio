"""Private real E01 native Wan2.2 full-episode producer, 7 I2V motion clips.

No independent SDXL shots, no InVideo/Runway, no public original art.
Strict source identity, native 24 fps, Russian Qwen3 WAVs already produced.
Hard-stop on first broken clip. Real Studio hybrid_master assembles MP4/SRT.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

CONFIG = "__EB01_V7_FULL_CONFIG_B64__"
STUDIO = Path("/kaggle/temp/eb01-v7-public-source")
COMFY = Path("/kaggle/temp/ComfyUI")
ROOT = Path("/kaggle/temp/eb01-v7-native-private")
WORK = Path("/kaggle/working")
CLIPS_OUT = WORK/"EB01_V7_REVIEW_REAL_CLIPS_PRIVATE"
RECEIPT = WORK/"EB01_V7_PRODUCTION_JOB.json"
FINAL = WORK/"EB01_V7_ELIZABETH_BONES_SOBESEDOVANIE_REAL_NATIVE_REVIEW.mp4"
FILM_QA = WORK/"EB01_V7_CREATIVE_REVIEW_REQUIRED.json"
MODEL = "wan22-5b"
# Frame refs from VERIFIED PRIVATE CPU source-mount Contact Sheet run #43186281307.
# Human-readable storyboard was inspected before authorizing any GPU inference.
REVIEWED_SOURCE_SHA={
 "S00":"7f1fef44ac5de0e51925024500058159bdf042f873ed3156f6a767001a5bd618",
 "S01":"9ce2cb7a35ddf4cf4c662b0b1d4917d989fb2c6f55491f54ff283af20b89a9a",
 "S02":"6f101608d5fd76adc85a976b775a5dc91612aa7a36b456c04660f859d7bc96a9",
 "S03":"16b076ec3fc9e3dbae87368940b06a8e444d0ebbf5e6d28a9ba10f7421547816",
 "S04":"6e366694363b69e88663a2fd3b82c2c93ea87ffeb6ac9d39675c1a776898120e",
 "S05":"eb5c4e86867141a4ebc442fa67550c8f10f4fa604c5805a3901e58b2e3841043",
 "S06":"5c622532d6ade6656869898364519124333d6964951a5c50c8082ef8b278c92d"
}

def command(args, *, timeout=1200, cwd=None):
    subprocess.run(args, cwd=cwd, check=True, timeout=timeout)

def authenticate_and_sources(cfg):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A genuine private T4 GPU is required")
    info = torch.cuda.get_device_properties(0)
    if info.total_memory < 14*(2**30):
        raise RuntimeError("GPU VRAM insufficient for explicitly selected Wan 5B")
    print("EB01_V7_TRUE_GPU",torch.cuda.get_device_name(0),flush=True)
    command(["git","clone","--depth","1","--branch",cfg["branch"],
             "https://github.com/19wolfalone92/Two-Worlds-Studio.git",str(STUDIO)],timeout=360)
    actual = subprocess.check_output(["git","rev-parse","HEAD"],cwd=STUDIO,text=True).strip()
    if actual != cfg["revision"]:
        raise RuntimeError("Public Studio producer revision changed after approval")
    sys.path.insert(0,str(STUDIO))
    from studio.eb01_v7_storyboard import SRC_SHA,VOICE_SHA,gather_private,SHOTS
    lock=gather_private(Path("/kaggle/input"),ROOT)
    if (lock.get("source_sha256") != SRC_SHA or
            lock.get("voice_sha256") != VOICE_SHA or
            list(lock.get("scenes",{})) != [v[0] for v in SHOTS]
            or {sid:row["reference_sha256"] for sid,row in lock["scenes"].items()} != REVIEWED_SOURCE_SHA):
        raise RuntimeError("Private reference/source image SHA differs from visually reviewed storyboard")
    print("EB01_V7_ALL_SIX_OLD_QWEN_WAV_PLUS_TWO_CANON_SOURCES_VERIFIED",flush=True)
    return info,lock

def neural_assets_and_server():
    command(["git","clone","--depth","1","https://github.com/Comfy-Org/ComfyUI.git",
             str(COMFY)],timeout=520)
    command([sys.executable,"-m","pip","install","-q","-r",str(COMFY/"requirements.txt"),
             "huggingface_hub>=0.29","Pillow>=10,<12","soundfile>=0.12",
             "numpy>=1.26,<3"],timeout=1500)
    from studio.model_install import install
    staged=install(MODEL,COMFY,accept_license=True,execute=True)
    if staged.get("status")!="WEIGHTS_STAGED_PRIVATELY_GPU_NOT_RUN":
        raise RuntimeError("Official Wan2.2 5B weights were not staged by native Studio installer")
    server_log=(WORK/"EB01_V7_COMFY_PRIVATE_SERVER.log").open("wb")
    server=subprocess.Popen(
        [sys.executable,"main.py","--listen","127.0.0.1","--port","8188",
         "--disable-auto-launch","--lowvram","--cpu-vae"],
        cwd=COMFY,stdout=server_log,stderr=subprocess.STDOUT)
    try:
        from studio.comfy_neural import ComfyLocal
        client=ComfyLocal("http://127.0.0.1:8188",timeout=25)
        start=time.monotonic()
        while time.monotonic()-start<300:
            if server.poll() is not None:
                raise RuntimeError("ComfyUI server stopped unexpectedly")
            try:
                client.json("/system_stats")
                return server,server_log,client,staged
            except Exception:
                time.sleep(3)
        raise TimeoutError("Real model Comfy server did not start")
    except BaseException:
        server.terminate()
        server_log.close()
        raise

def sample_frames(clip:Path,first:Path,last:Path,frames:int):
    command(["ffmpeg","-v","error","-nostdin","-i",str(clip),
             "-vf","select=eq(n\\,0)","-vsync","vfr","-frames:v","1",str(first)],timeout=90)
    command(["ffmpeg","-v","error","-nostdin","-i",str(clip),
             "-vf","select=eq(n\\,"+str(frames-1)+")","-vsync","vfr","-frames:v","1",str(last)],timeout=90)
    if not first.is_file() or not last.is_file():
        raise RuntimeError("Could not decode both ends of real motion clip")
    from PIL import Image,ImageChops,ImageStat
    with Image.open(first) as a,Image.open(last) as b:
        change=float(sum(ImageStat.Stat(ImageChops.difference(a.convert("RGB"),b.convert("RGB"))).mean)/3)
    return change

def generate_clips(cfg,lock,client):
    from studio.comfy_neural import generate
    from studio.eb01_v7_storyboard import SHOTS,WIDTH,HEIGHT,FPS,sha
    from studio.eb01_motion_artifact_gate import inspect_clip
    from studio.eb01_v7_motion_buffer import finalize
    import torch
    CLIPS_OUT.mkdir(parents=True,exist_ok=False)
    (ROOT/"clips").mkdir(exist_ok=True)
    contact=[]
    results={}
    for number,(sid,who,frames,duration,zoom,cx,cy,prompt) in enumerate(SHOTS):
        src=ROOT/"frames"/(sid+".png")
        if sha(src)!=lock["scenes"][sid]["reference_sha256"]:
            raise RuntimeError("Episode frame changed since canonical SHA lock "+sid)
        movie=ROOT/"clips"/(sid+".mp4")
        raw_movie=ROOT/"clips"/(sid+".raw.mp4")
        print("EB01_REAL_MOTION_START",sid,who,
              "deliver_real_frames",frames,"raw_with_4_extra_frames",frames+4,
              "seconds",round(frames/FPS,3),flush=True)
        start=time.monotonic()
        made=generate(MODEL,reference=src,output=raw_movie,prompt=prompt,
                      seed=202611120+number*31,host="http://127.0.0.1:8188",
                      width=WIDTH,height=HEIGHT,frames=frames+4,steps=20,
                      timeout=10600,execute=True,accepted_license=True,client=client)
        if made.get("status")!="GENERATED_REVIEW_REQUIRED":
            raise RuntimeError("Wan2.2 did not create a genuine native I2V video for "+sid)
        # The model's last 3 frames once developed neon stripes. Four spare
        # frames are real model output; do not pad/freeze useful performance.
        # Fail if ANY glitch occurs inside the actual user's performance window.
        lineage=finalize(raw_movie,raw_movie.with_suffix(".receipt.json"),
                         src,movie,frames)
        audit=inspect_clip(movie)
        if audit["status"]!="NO_NEON_ARTIFACT_FOUND" or audit["decoded_frames"]!=frames:
            raise RuntimeError("Temporal corrupt/stripe footage forbidden at "+sid+
                               ": "+audit["status"])
        start_frame=ROOT/(sid+"_first.png")
        end_frame=ROOT/(sid+"_last.png")
        change=sample_frames(movie,start_frame,end_frame,frames)
        if change<.80:
            raise RuntimeError("Full video frame frozen/static; do NOT waste GPU on next shots "+sid)
        proof=movie.with_suffix(".receipt.json")
        data=json.loads(proof.read_text(encoding="utf-8"))
        if (data.get("reference_sha256")!=sha(src)
                or data.get("clip_sha256")!=sha(movie)
                or data.get("inference_performed") is not True
                or data.get("model")!=MODEL
                or data.get("video",{}).get("frames")!=frames):
            raise RuntimeError("Model output has no trustworthy Comfy provenance "+sid)
        # Preserve each verified clip on PRIVATE Kaggle even if later shots fail.
        dest=CLIPS_OUT/(sid+".mp4")
        dest.write_bytes(movie.read_bytes())
        (CLIPS_OUT/(sid+".receipt.json")).write_bytes(proof.read_bytes())
        r={"id":sid,"character":who,"real_motion":True,"model":MODEL,
           "requested_native_frames":frames,"decoded_native_frames":frames,
           "runtime_seconds":round(time.monotonic()-start,1),
           "clip_sha256":sha(movie),"reference_sha256":sha(src),
           "motion_first_last_difference":round(change,4),
           "neon_artifact_qc":audit["status"],
           "four_frame_source_buffer_provenance":lineage,
           "human_visual_reviewed":False}
        results[sid]=r
        (WORK/"EB01_V7_PROGRESS_PRIVATE.json").write_text(
            json.dumps({"schema":7,"status":"RUNNING_REAL_NEURAL_SHOTS",
                        "completed_shots":results,"published":False},indent=2))
        print("EB01_V7_NATIVE_SHOT_PASSED_CPU_CHECK_PENDING_HUMAN",sid,
              r["clip_sha256"],r["runtime_seconds"],flush=True)
        torch.cuda.empty_cache()
    return results

def create_score(folder:Path,seconds:float):
    import numpy as np, soundfile as sf
    sr=24000
    t=np.arange(math.ceil(seconds*sr),dtype=np.float64)/sr
    score=np.zeros_like(t)
    for hz,amp in ((55.0,.045),(82.41,.034),(110.0,.022),(146.83,.009)):
        score+=amp*np.sin(t*2*math.pi*hz)
    # A soft minor-key pulsing room score, no third-party music copied.
    fade=np.minimum(1,np.minimum(t/1.3,(seconds-t)/2.0))
    score=score*np.maximum(0,fade)
    sf.write(folder/"score.wav",score.astype(np.float32),sr)

def assemble_native(cfg,lock,real_clips):
    from studio.eb01_v7_storyboard import SHOTS,DIALOGUES,VOICE_SECONDS,VOICE_SHA,sha,FPS
    import soundfile as sf
    from studio.hybrid_master import produce
    firsts=[]
    for sid,who,frames,seconds,zoom,cx,cy,prompt in SHOTS:
        clip=ROOT/"clips"/(sid+".mp4")
        proof=ROOT/"clips"/(sid+".receipt.json")
        ref=ROOT/"frames"/(sid+".png")
        if not clip.is_file() or sha(clip)!=real_clips[sid]["clip_sha256"]:
            raise RuntimeError("Full film requires seven original I2V clips")
        row={"clip":"clips/"+sid+".mp4",
             "duration":seconds,
             "shot_type":"establishing" if sid=="S00" else "medium",
             "camera":"push_in","camera_strength":0.0,
             "caption_position":"bottom","caption":DIALOGUES.get(sid,""),
             "approved":False,
             "clip_take":{"engine":"neural_comfy","visual_profile":"photoreal_cinematic",
                          "sha256":sha(clip),"receipt":"clips/"+sid+".receipt.json",
                          "reference":"frames/"+sid+".png"}}
        if sid in VOICE_SHA:
            wav=ROOT/"voices"/(sid+".wav")
            with sf.SoundFile(wav) as f:
                duration=f.frames/f.samplerate
                if f.channels!=1 or abs(duration-VOICE_SECONDS[sid])>.025:
                    raise RuntimeError("Original Qwen voice clock mismatch "+sid)
            if sha(wav)!=VOICE_SHA[sid] or seconds<duration+.08:
                raise RuntimeError("Real audio could not fit genuine motion clip "+sid)
            row.update({"voice":"voices/"+sid+".wav",
                        "voice_bus":"dialogue","voice_offset":.07,
                        "voice_approved":False,"dialogue_text":DIALOGUES[sid]})
        firsts.append(row)
    total=sum(row["duration"] for row in firsts)
    create_score(ROOT,total+.4)
    manifest={"schema":1,"delivery_profile":"mobile",
              "width":576,"height":1024,"fps":FPS,
              "language":"ru","visual_profile":"photoreal_cinematic",
              "burn_subtitles":True,"music":"score.wav","music_volume":.095,
              "shots":firsts,"editorial_approved":False,"rights_approved":False}
    path=ROOT/"native_eb01_v7_manifest.json"
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    report=produce(path,FINAL,master=False,quality_check=True)
    if (report["shots"]!=7 or report.get("static_camera_shots")!=0
            or report.get("motion_clips")!=7
            or report.get("audio_tracks",0)<7
            or report.get("published") is not False):
        raise RuntimeError("Native Studio rendered incomplete or still-only fake movie")
    from studio.hybrid_master import probe
    meta=probe(FINAL)
    stream=meta.get("streams",[])
    if len([s for s in stream if s.get("codec_type")=="video"])!=1 or len([s for s in stream if s.get("codec_type")=="audio"])!=1:
        raise RuntimeError("Full film missing video or original Russian audio")
    srt=FINAL.with_suffix(".srt")
    content=srt.read_text(encoding="utf-8")
    if any(line not in content for line in DIALOGUES.values()) or content.count("-->")!=6:
        raise RuntimeError("Six correct Russian subtitle cues missing from film")
    command(["ffmpeg","-hide_banner","-nostdin","-loglevel","error","-xerror","-i",str(FINAL),
             "-f","null","-"],timeout=260)
    doc={"schema":7,"status":"FULL_NATIVE_MP4_TECHNICAL_QA_PASSED_ART_REVIEW_REQUIRED",
         "source_model":MODEL,
         "native_studio":"studio.hybrid_master.produce","invideo_used":False,
         "real_I2V_shots":7,"independent_text_to_image_shots":0,
         "russian_voice_lines":6,"subtitles":6,
         "full_film_sha256":sha(FINAL),
         "full_film_duration_seconds":report["duration_seconds"],
         "source_private_canon_sha256":lock["source_sha256"],
         "sources":real_clips,
         "video_exists":FINAL.is_file(),
         "owner_artistic_approved":False,
         "human_dialogue_lipsync_approved":False,
         "published":False}
    FILM_QA.write_text(json.dumps(doc,ensure_ascii=False,indent=2))
    print("EB01_V7_FULL_REAL_NATIVE_REVIEW_MP4_CREATED",doc["full_film_sha256"],
          doc["full_film_duration_seconds"],flush=True)
    return doc

def main():
    import base64
    cfg=json.loads(base64.b64decode(CONFIG,validate=True))
    server=None;server_log=None
    try:
        if (cfg.get("schema")!=7 or cfg.get("model")!=MODEL or
                cfg.get("private") is not True or cfg.get("no_paid_services") is not True
                or cfg.get("allow_social_publish") is not False
                or cfg.get("image_sources")!=2 or cfg.get("voice_lines")!=6
                or cfg.get("scene_count")!=7):
            raise ValueError("Unreviewed full episode GPU request blocked")
        gpu,lock=authenticate_and_sources(cfg)
        server,server_log,client,staged=neural_assets_and_server()
        real=generate_clips(cfg,lock,client)
        # Server no longer needed for FFmpeg CPU mastering.
        server.terminate()
        server.wait(timeout=25)
        server_log.close()
        server=None
        proof=assemble_native(cfg,lock,real)
        RECEIPT.write_text(json.dumps({
            "schema":7,"success":True,"status":proof["status"],
            "run_nonce":cfg["nonce"],
            "source_revision":cfg["revision"],"gpu":gpu.name,
            "model":MODEL,"seven_native_clips":True,
            "full_mp4_name":FINAL.name,
            "full_mp4_sha256":proof["full_film_sha256"],
            "human_approved":False,"published":False
        },indent=2))
    except BaseException as e:
        RECEIPT.write_text(json.dumps({
            "schema":7,"success":False,"status":"BLOCKED",
            "run_nonce":cfg.get("nonce"),"error_type":type(e).__name__,
            "error_summary":str(e)[:400],
            "partial_neural_clips_may_be_present":CLIPS_OUT.is_dir(),
            "published":False},ensure_ascii=False,indent=2))
        print("EB01_V7_FULL_NEURAL_BLOCKED",type(e).__name__,str(e)[:400],flush=True)
        traceback.print_exc()
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=20)
            except Exception:server.kill()
        if server_log is not None and not server_log.closed:
            server_log.close()

if __name__=="__main__":
    main()
