"""CPU-only release candidate composer for seven authentic PRIVATE Wan I2V takes.

No ComfyUI, Kaggle GPU, fixture, random art or paid service. Collects original
S00/S01 from CANCEL_ACKNOWLEDGED Kaggle job, individual PRIVATE S02..S06,
two authentic photo sources and six original Russian Qwen WAVs from a
PRIVATE dataset. Rejects missing/duplicate/mutated clips BEFORE production.
An owner-review MP4 is NOT declared artistically approved automatically.
"""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from studio.eb01_v7_storyboard import SHOTS,SRC_SHA,VOICE_SHA,gather_private,sha
from studio.eb01_motion_artifact_gate import inspect_clip
from studio.comfy_neural import _probe

OLD_GENUINE_SHA={
    "S00":"7ac34976cfc5086ad068bffd837893a3c41ee6a5d48cc1fcd66f50c368e8b208",
    "S01":"fe0e45802d6df19af951dc1ebb8cfaeb396fdcce59d860d5826a62a7e7d36c87",
}
REQUIRED_SHOTS={row[0] for row in SHOTS}


class EB01RecoveryBlocked(RuntimeError):
    pass


def verify_real_clips(staging:Path, clip_sources:dict[str,Path], lock:dict)->dict:
    """Require every real I2V source clip+receipt exactly once, all seven."""
    if set(clip_sources)!=REQUIRED_SHOTS:
        raise EB01RecoveryBlocked("Exactly original S00,S01 plus five later real shots required")
    output=staging/"clips"
    output.mkdir(exist_ok=False)
    rows={}
    for sid,who,frames,seconds,*_ in SHOTS:
        root=Path(clip_sources[sid])
        matches=list(root.rglob(sid+".mp4"))
        receipts=list(root.rglob(sid+".receipt.json"))
        if len(matches)!=1 or len(receipts)!=1:
            raise EB01RecoveryBlocked("Missing or duplicate genuine scene or Comfy receipt "+sid)
        mp4=matches[0]
        proof=json.loads(receipts[0].read_text(encoding="utf-8"))
        actual_sha=sha(mp4)
        if (proof.get("engine")!="neural_comfy"
                or proof.get("model")!="wan22-5b"
                or proof.get("inference_performed") is not True
                or proof.get("reference_sha256")!=lock["scenes"][sid]["reference_sha256"]
                or proof.get("clip_sha256")!=actual_sha
                or proof.get("human_approved") is not False
                or proof.get("published") is not False):
            raise EB01RecoveryBlocked("No trustworthy private genuine Comfy I2V source for "+sid)
        if sid in OLD_GENUINE_SHA:
            if actual_sha!=OLD_GENUINE_SHA[sid]:
                raise EB01RecoveryBlocked("Original rescued scene has changed: "+sid)
        else:
            jobs=list(root.rglob("EB01_V7_SINGLE_SHOT_JOB.json"))
            if len(jobs)!=1:
                raise EB01RecoveryBlocked("New isolated GPU scene has no independent receipt "+sid)
            job=json.loads(jobs[0].read_text())
            if (job.get("status")!="ONE_REAL_WAN_SHOT_TECHNICALLY_VERIFIED"
                or job.get("success") is not True
                or job.get("shot")!=sid
                or job.get("clip_sha256")!=actual_sha
                or job.get("frame_count")!=frames
                or job.get("neural_inference_performed") is not True
                or job.get("original_reference_sha256")!=proof["reference_sha256"]
                or job.get("temporal_rgb_artifact_gate")!="NO_NEON_ARTIFACT_FOUND"):
                raise EB01RecoveryBlocked("Missing verified independent one-scene private job "+sid)
        # Actual model-generated video, not green CI; independently ffprobe,
        # inspect EVERY RGB frame and decode MP4 fully.
        meta=_probe(mp4)
        if meta.get("frames")!=frames or meta.get("fps")!="24":
            raise EB01RecoveryBlocked("Real Wan output frames/FPS mismatch: "+sid)
        qc=inspect_clip(mp4)
        if qc["status"]!="NO_NEON_ARTIFACT_FOUND" or qc["decoded_frames"]!=frames:
            raise EB01RecoveryBlocked("Temporal model artefact detected in "+sid)
        run=subprocess.run(["ffmpeg","-hide_banner","-nostdin","-v","error",
                            "-xerror","-i",str(mp4),"-f","null","-"],
                           text=True,capture_output=True,timeout=180)
        if run.returncode:
            raise EB01RecoveryBlocked("Real H264 video cannot completely decode: "+sid)
        shutil.copy2(mp4,output/(sid+".mp4"))
        shutil.copy2(receipts[0],output/(sid+".receipt.json"))
        rows[sid]={
            "character":who,"real_model":"wan22-5b",
            "private_model_source_sha256":actual_sha,
            "original_reference_sha256":proof["reference_sha256"],
            "frames":frames,"fps":24,"ffmpeg_fully_decoded":True,
            "color_artifact_gate":"NO_NEON_ARTIFACT_FOUND",
            "published":False,"human_art_approved":False,
        }
    return rows


def produce_review(dataset_root:Path, clip_sources:dict[str,Path],
                   workspace:Path, output:Path)->dict:
    dataset_root,workspace,output=map(Path,(dataset_root,workspace,output))
    if output.exists() or output.is_symlink():
        raise EB01RecoveryBlocked("Never overwrite a movie: must create a new review")
    if workspace.exists():
        raise EB01RecoveryBlocked("Must use new/private ephemeral assembly workspace")
    lock=gather_private(dataset_root,workspace)
    if lock.get("source_sha256")!=SRC_SHA or lock.get("voice_sha256")!=VOICE_SHA:
        raise EB01RecoveryBlocked("Original character/voice source lock mismatch")
    shot_details=verify_real_clips(workspace,clip_sources,lock)
    from tools import kaggle_eb01_v7_full_entry as composer
    composer.ROOT=workspace
    composer.FINAL=output
    composer.FILM_QA=output.with_name("EB01_V7_RESUME_NATIVE_REVIEW_QA.json")
    film=composer.assemble_native(
        {"private":True},lock,{k:{"clip_sha256":v["private_model_source_sha256"]}
                                for k,v in shot_details.items()})
    if (film.get("real_I2V_shots")!=7 or film.get("russian_voice_lines")!=6
        or film.get("subtitles")!=6 or film.get("invideo_used") is not False
        or film.get("native_studio")!="studio.hybrid_master.produce"
        or film.get("full_film_sha256")!=sha(output)):
        raise EB01RecoveryBlocked("Actual native full movie result failed provenance gate")
    details={
        "schema":7,
        "status":"SEVEN_REAL_NATIVE_SHOTS_PLUS_SIX_ORIGINAL_VOICES_REVIEW_REQUIRED",
        "output_file":output.name,
        "output_sha256":sha(output),"true_neural_takes":shot_details,
        "dialogue_lines":6,"srt_cues":6,
        "voice_original_sha256":VOICE_SHA,
        "canon_original_sha256":SRC_SHA,
        "native_editor":"studio.hybrid_master.produce",
        "invideo_used":False,
        "all_frames_source_genuine":True,
        "human_artistic_review_completed":False,
        "published":False,
    }
    receipt=output.with_name("EB01_V7_RECOVERED_MASTER_RECEIPT.json")
    receipt.write_text(json.dumps(details,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return details
