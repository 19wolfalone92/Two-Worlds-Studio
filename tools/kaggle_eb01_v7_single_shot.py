"""One real native Wan2.2 EB01 shot per PRIVATE Kaggle T4 session.

EB01 original full seven-shot notebook was CANCEL_ACKNOWLEDGED during S02,
after S00 and S01. The pipeline must never run all seven long I2V shots in
one 3600-second Kaggle job. This worker only creates ONE missing shot,
with original input image/voice SHA pinned, genuine ComfyUI model provenance,
and strict 4-frame neon-tail buffer QC.

No retries, no fixture mode, no independent text-to-image regeneration,
no InVideo, no paid API and no public source art.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

CONFIG_B64="__EB01_V7_RESUME_SHOT_CONFIG_B64__"
ROOT=Path("/kaggle/temp/eb01-v7-public-source")
WORK=Path("/kaggle/working")
RECEIPT=WORK/"EB01_V7_SINGLE_SHOT_JOB.json"
ALLOWED={"S02","S03","S04","S05","S06"}

def digest(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def check_configuration(config:dict)->str:
    if (config.get("schema")!=7
            or config.get("shot") not in ALLOWED
            or config.get("model")!="wan22-5b"
            or config.get("private") is not True
            or config.get("new_gpu_jobs")!=1
            or config.get("allow_social_publish") is not False
            or config.get("allow_regenerate_good_shots") is not False
            or config.get("source_dataset")!="wolf19loner92/eb01-v7-private-production-sources"
            or not isinstance(config.get("revision"),str)
            or len(config["revision"])!=40):
        raise ValueError("Invalid or unsafe isolated real I2V shot request")
    # Each artistic retake is a separate, immutable private Kaggle kernel.
    # Only rejected S02/S05 or a *failed* new S06 is eligible; no changes
    # to the already verified S00/S01/S03/S04 source media.
    retake=config.get("quality_retake",False)
    attempt=config.get("seed_variant",0)
    if (type(retake) is not bool or type(attempt) is not int
            or attempt not in (0,1,2)
            or (retake and config["shot"] not in ("S02","S05","S06"))
            or (attempt>0 and not retake)
            or (config["shot"]=="S02" and not retake)):
        raise ValueError("Unsafe or unbounded native Wan quality-retake request")
    return config["shot"]

def stage_reproducible_source(cfg):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Real T4 GPU missing")
    if torch.cuda.get_device_properties(0).total_memory<14*2**30:
        raise RuntimeError("Need >=14GiB GPU, no CPU mock/fixture")
    subprocess.run(["git","clone","--depth","1","--branch",cfg["branch"],
                    "https://github.com/19wolfalone92/Two-Worlds-Studio.git",str(ROOT)],
                   check=True,timeout=360)
    subprocess.run(["git","fetch","--depth","1","origin",cfg["revision"]],
                   cwd=ROOT,check=True,timeout=240)
    subprocess.run(["git","checkout","--detach",cfg["revision"]],
                   cwd=ROOT,check=True,timeout=60)
    actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if actual!=cfg["revision"]:
        raise RuntimeError("Source revision not pinned")
    sys.path.insert(0,str(ROOT))
    from studio.eb01_v7_storyboard import gather_private,SRC_SHA,VOICE_SHA,SHOTS
    from tools.kaggle_eb01_v7_full_entry import REVIEWED_SOURCE_SHA
    store=Path("/kaggle/temp/eb01-v7-resume-private")
    lock=gather_private(Path("/kaggle/input"),store)
    if (lock["source_sha256"]!=SRC_SHA or lock["voice_sha256"]!=VOICE_SHA
            or {k:v["reference_sha256"] for k,v in lock["scenes"].items()}!=REVIEWED_SOURCE_SHA):
        raise RuntimeError("Seven canon references do not match original private review")
    print("EB01_V7_RESUME_ORIGINAL_PRIVATE_SOURCE_SHA_VERIFIED",cfg["shot"],flush=True)
    return store,lock

def generate_one(cfg,store,lock):
    from tools import kaggle_eb01_v7_full_entry as full
    from studio.eb01_v7_storyboard import SHOTS,WIDTH,HEIGHT,FPS,sha
    from studio.eb01_motion_artifact_gate import inspect_clip
    from studio.eb01_v7_motion_buffer import finalize, MotionBufferRejected
    from studio.eb01_v7_private_checkpoint import (
        checkpoint_raw, mark_quarantined, selected_contact_sheet
    )
    from studio.comfy_neural import generate
    import torch

    name=cfg["shot"]
    row=next((r for r in SHOTS if r[0]==name),None)
    if row is None:
        raise RuntimeError("S00/S01 original shots are already complete and must NEVER be regenerated")
    sid,who,frames,seconds,zoom,cx,cy,prompt=row
    path=store/"frames"/(sid+".png")
    if sha(path)!=lock["scenes"][sid]["reference_sha256"]:
        raise RuntimeError("Original single-scene SHA mismatch")
    # Official model setup is the same native Comfy worker that generated
    # genuine S00/S01. Only one shot now: no global 7-clip time budget.
    server=None
    server_log=None
    try:
        server,server_log,client,staged=full.neural_assets_and_server()
        raw=store/(sid+".raw.mp4")
        final=WORK/(sid+".mp4")
        if final.exists():
            raise RuntimeError("Never overwrite an existing good scene")
        print("EB01_V7_SINGLE_NATIVE_I2V_INFERENCE_START",sid,who,
              "raw_frames",frames+4,"kept_frames",frames,flush=True)
        start=time.monotonic()
        # This is a separately reviewed recovery, not deterministic replay
        # of the same 129-frame failed candidate. One new seed only.
        chosen_seed=(202611130+int(sid[1:])*31
                     if sid=="S02" else 202611120+int(sid[1:])*31)
        # Rejected S02/S05 clips exhibit broad neon latent corruption, not
        # an encoding error. Do not recycle the proven-bad stochastic seed.
        # Retakes use a genuinely different diffusion trajectory and a
        # modestly longer sampler. Never relabel an unchanged RAW as fixed.
        variant=cfg.get("seed_variant",0)
        if sid=="S02" and variant==0:
            raise RuntimeError("Known-neon S02 requires explicit new diffusion seed")
        chosen_seed += 1009*variant
        sample_steps=(20 if variant==0 else (24 if variant==1 else 28))
        print("EB01_WAN_QUALITY_RETAKE_CONFIG",sid,"variant",variant,
              "seed",chosen_seed,"steps",sample_steps,flush=True)
        made=generate(full.MODEL,reference=path,output=raw,prompt=prompt,
                      seed=chosen_seed,
                      host="http://127.0.0.1:8188",
                      width=WIDTH,height=HEIGHT,frames=frames+4,
                      steps=sample_steps,timeout=10600,execute=True,
                      accepted_license=True,client=client)
        if made.get("status")!="GENERATED_REVIEW_REQUIRED":
            raise RuntimeError("No real native Wan inference returned for "+sid)
        # CRITICAL: genuine Wan source and original model provenance MUST survive
        # any artistic/temporal QC rejection or Kaggle worker termination.
        # Old S02 rendered 129 frames but lost the private MP4 after QC failed
        # because the raw file remained only under /kaggle/temp.
        original=checkpoint_raw(
            raw, raw.with_suffix(".receipt.json"), path, WORK, sid, frames
        )
        print("EB01_REAL_WAN_ORIGINAL_SAVED_PRIVATE_BEFORE_QC",sid,
              original["original_mp4_sha256"],flush=True)
        try:
            lineage=finalize(raw,raw.with_suffix(".receipt.json"),path,final,frames)
        except MotionBufferRejected as failure:
            # Bright saturated pixels may be intentional lighting, so the
            # heuristic is a REVIEW FLAG, not proof of corrupt cinema.
            # Never silently approve; preserve RAW bytes + contact frame JPG.
            try:
                scan=inspect_clip(raw)
                flagged=scan.get("flagged_frames",[])
                inspection={
                    "status":scan.get("status"),
                    "flagged_frames":flagged,
                    "threshold":scan.get("threshold"),
                    "frame_saturation_trace":scan.get("frame_saturation_trace",[]),
                }
                mark_quarantined(WORK,sid,str(failure),inspection=inspection)
                if flagged:
                    chosen=sorted(set([0,1,10,max(0,flagged[0]-1)] +
                                      flagged[:14] + [min(frames+3,flagged[-1]+1)]))
                    chosen=chosen[:20]
                    selected_contact_sheet(
                        WORK/(sid+".original_wan_raw.mp4"),
                        WORK/(sid+".LOW_RES_FRAME_REVIEW_ONLY.jpg"),chosen
                    )
            except Exception as review_failure:
                print("EB01_PRIVATE_CONTACT_REVIEW_ERROR_NONFATAL",
                      type(review_failure).__name__,
                      str(review_failure)[:180],flush=True)
            print("EB01_RAW_PERSISTED_BUT_ARTISTICALLY_QUARANTINED",sid,
                  str(failure)[:350],flush=True)
            raise
        qc=inspect_clip(final)
        if qc["status"]!="NO_NEON_ARTIFACT_FOUND" or qc["decoded_frames"]!=frames:
            raise RuntimeError("Model video contains bad/interrupted RGB frames")
        first,last=store/(sid+"_first.png"),store/(sid+"_last.png")
        motion=full.sample_frames(final,first,last,frames)
        if motion<.8:
            raise RuntimeError("Static film frame is not acceptable native I2V")
        proof=json.loads(final.with_suffix(".receipt.json").read_text(encoding="utf-8"))
        if (proof.get("clip_sha256")!=digest(final)
                or proof.get("reference_sha256")!=sha(path)
                or proof.get("inference_performed") is not True):
            raise RuntimeError("Generated model/actor clip proof is incomplete")
        result={
            "schema":7,"status":"ONE_REAL_WAN_SHOT_TECHNICALLY_VERIFIED",
            "success":True,"shot":sid,"actor":who,"nonce":cfg["nonce"],
            "source_revision":cfg["revision"],
            "model":"wan22-5b","native_studio":"studio.comfy_neural.generate",
            "neural_inference_performed":True,"invideo_used":False,
            "original_character_sha256":lock["source_sha256"][who],
            "original_reference_sha256":sha(path),
            "clip_sha256":digest(final),
            "clip_filename":final.name,
            "frame_count":frames,"fps":FPS,
            "width":WIDTH,"height":HEIGHT,
            "duration_seconds":frames/FPS,
            "native_original_source_provenance":lineage,
            "original_raw_private_checkpoint":original,
            "inference_seed":chosen_seed,
            "motion_first_last_mean_difference":motion,
            "temporal_rgb_artifact_gate":qc["status"],
            "seconds_elapsed":round(time.monotonic()-start,2),
            "human_artistic_approved":False,
            "full_episode_ready":False,"published":False
        }
        print("EB01_V7_SINGLE_REAL_SHOT_SAVED_PRIVATE",sid,
              result["clip_sha256"],round(result["duration_seconds"],2),flush=True)
        return result
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=15)
            except Exception:server.kill()
        if server_log is not None and not server_log.closed:
            server_log.close()

def main():
    cfg=json.loads(base64.b64decode(CONFIG_B64,validate=True))
    sid=check_configuration(cfg)
    try:
        store,lock=stage_reproducible_source(cfg)
        result=generate_one(cfg,store,lock)
        RECEIPT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    except BaseException as exc:
        # The original generated MP4 now survives ANY aesthetic rejection.
        # Report this distinction so collectors do not confuse true model
        # inference with a failed/empty Kaggle job or an approved scene.
        checkpoint=WORK/(sid+".original_wan_raw.checkpoint.json")
        checkpoint_data={}
        if checkpoint.is_file():
            try:
                checkpoint_data=json.loads(checkpoint.read_text(encoding="utf-8"))
            except Exception:
                pass
        preserved=checkpoint_data.get("status")=="GENUINE_NEURAL_SOURCE_PRESERVED_QA_PENDING"
        RECEIPT.write_text(json.dumps({
            "schema":7,
            "status":("REAL_GPU_VIDEO_PRESERVED_BUT_VISUAL_QA_FAILED"
                      if preserved else "INCOMPLETE_STOP_REQUIRES_DIAGNOSIS"),
            "success":False,"shot":sid,"nonce":cfg["nonce"],
            "error_type":type(exc).__name__,
            "error_summary":str(exc)[:350],
            "model_inference_performed":preserved,
            "private_raw_mp4_preserved":preserved,
            "private_raw_mp4_sha256":checkpoint_data.get("original_mp4_sha256"),
            "needs_visual_review":preserved,
            "finished_episode_ready":False,
            "no_auto_retry":True,"published":False},
            ensure_ascii=False,indent=2)+"\n")
        print("EB01_V7_SINGLE_SHOT_ERROR",sid,type(exc).__name__,
              str(exc)[:350],"raw_model_video_persisted",preserved,flush=True)
        traceback.print_exc()
        raise
        print("EB01_V7_SINGLE_SHOT_ERROR",sid,type(exc).__name__,
              str(exc)[:350],flush=True)
        traceback.print_exc()
        raise

if __name__=="__main__":
    main()
