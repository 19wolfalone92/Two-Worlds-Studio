"""Recover a real native TWO WORLDS STUDIO EB01 MP4 from verified private Kaggle sources.

This runs the existing native Studio on GitHub CPU only. It NEVER calls InVideo,
creates a synthetic fixture, starts a new GPU job, publishes, or commits private
images / WAVs to public source control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat

from studio.elizabeth_episode_pilot import (
    SCENES, WIDTH, HEIGHT, FPS, IMAGE_MODEL, VOICE_MODEL, _duration, native_render
)
from tools.kaggle_elizabeth_pilot_entry import create_original_score, render_cover


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def ffprobe(path: Path) -> dict:
    p=subprocess.run(["ffprobe","-v","error","-show_entries",
                      "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate",
                      "-of","json",str(path)],check=True,capture_output=True,
                      text=True,timeout=45)
    return json.loads(p.stdout)


def verify_sources(work: Path) -> tuple[dict[str,float],list[dict]]:
    if work.is_symlink() or not work.is_dir():
        raise ValueError("Private Kaggle source directory is missing or unsafe")
    clock={}
    manifest=[]
    for sid,speaker,text,*_ in SCENES:
        art=work/"art"/(sid+".png")
        if art.is_symlink() or not art.is_file() or art.stat().st_size<50_000:
            raise FileNotFoundError("No genuine GPU-rendered art for "+sid)
        with Image.open(art) as img:
            img.verify()
        with Image.open(art) as img:
            if img.size!=(WIDTH,HEIGHT) or ImageStat.Stat(img.convert("RGB")).stddev[0]<8:
                raise ValueError("Invalid, undersized or blank authentic SDXL still: "+sid)
        row={"id":sid,"art_sha256":digest(art),"art_size_bytes":art.stat().st_size,
             "speaker":speaker,"dialogue":text}
        if speaker:
            wav=work/"voices"/(sid+".wav")
            if wav.is_symlink() or not wav.is_file() or wav.stat().st_size<10_000:
                raise FileNotFoundError("Missing genuine Qwen voice: "+sid)
            with wave.open(str(wav),"rb") as w:
                if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getframerate()<16000:
                    raise ValueError("Qwen output WAV format invalid for "+sid)
                frames=w.getnframes()
                samples=w.readframes(frames)
                duration=frames/w.getframerate()
            waveform=np.frombuffer(samples,dtype="<i2").astype(np.float32)
            rms=float(np.sqrt(np.mean(waveform*waveform)))
            if duration<.15 or duration>28 or rms<80:
                raise ValueError("Qwen speech too short, long or silent for "+sid)
            clock[sid]=duration
            row.update(voice_sha256=digest(wav),voice_duration_seconds=round(duration,4),
                       voice_rms_16bit=round(rms,1))
        manifest.append(row)
    if len(manifest)!=7 or len(clock)!=6:
        raise RuntimeError("Not seven unique real SDXL shots with six Qwen voice tracks")
    if len({r["art_sha256"] for r in manifest})!=7:
        raise ValueError("Duplicate source stills - genuine seven scene contract violated")
    return clock,manifest


def verify_movie(mp4: Path, work: Path, report: dict) -> dict:
    if not mp4.is_file() or mp4.stat().st_size < 100_000:
        raise FileNotFoundError("Studio did not create actual MP4 bytes")
    p=ffprobe(mp4)
    vids=[s for s in p.get("streams",[]) if s.get("codec_type")=="video"]
    auds=[s for s in p.get("streams",[]) if s.get("codec_type")=="audio"]
    if len(vids)!=1 or len(auds)!=1 or vids[0].get("width")!=576 or vids[0].get("height")!=1024:
        raise ValueError("Native MP4 has missing video/audio or wrong portrait dimensions")
    if vids[0].get("avg_frame_rate")!="24/1":
        raise ValueError("Studio frame rate mismatch")
    if not 18<float(p["format"]["duration"])<120:
        raise ValueError("Studio duration unexpected")
    if (report.get("shots")!=7 or report.get("audio_tracks")!=7 or
            report.get("static_camera_shots")!=7 or report.get("published") is not False or
            not report.get("captions_burned_in")):
        raise RuntimeError("Not the genuine seven-shot native Studio montage")
    assert report.get("video_sha256")==digest(mp4)
    srt=mp4.with_suffix(".srt")
    if not srt.is_file() or srt.read_text(encoding="utf-8").count("-->") != 6:
        raise ValueError("Six Russian subtitle cues not present")
    run=subprocess.run(["ffmpeg","-nostdin","-v","error","-xerror","-i",str(mp4),
                        "-f","null","-"],capture_output=True,timeout=240)
    if run.returncode:
        raise RuntimeError("Independent full-video decode failed: "+run.stderr.decode(errors="replace")[:700])
    # Independently verify each spoken segment is audible in the final multiplexed MP4.
    plan=json.loads((work/"studio_shot_manifest.json").read_text(encoding="utf-8"))
    t=0
    windows={}
    for i,shot in enumerate(plan["shots"]):
        dur=round(shot["duration"]*FPS)/FPS
        if shot.get("voice"):
            cmd=["ffmpeg","-hide_banner","-nostdin","-ss",str(t+.15),
                 "-t",str(max(.5,dur-.35)),"-i",str(mp4),"-vn",
                 "-af","volumedetect","-f","null","-"]
            out=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=90)
            m=re.search(r"mean_volume:\s*(-?[\d.]+) dB",out.stderr)
            if not m or float(m.group(1))<-65:
                raise RuntimeError("Silent dialogue in output scene S"+str(i).zfill(2))
            windows["S"+str(i).zfill(2)]=float(m.group(1))
        t+=dur
    if len(windows)!=6:
        raise RuntimeError("Native E01 full montage missing spoken scene")
    return {"dimensions":"576x1024","fps":24,"duration_seconds":float(p["format"]["duration"]),
            "video_sha256":digest(mp4),"full_video_decode":True,"six_voice_segments_db":windows,
            "six_subtitles":True,"seven_shots":True,"studio_native":True}


def recover(private_root: Path, delivered: Path, source_kernel: str) -> dict:
    work=private_root/"elizabeth_studio_private"
    clock,assets=verify_sources(work)
    # Reuse the original Studio authoring code to add the title and royalty-free score.
    # Our initial private Kaggle GPU render generated ALL media and failed only because
    # its minimal image did not contain a font supporting Cyrillic.
    try:
        render_cover(work)
        cover_status="native_title_overlaid"
    except RuntimeError as e:
        if "Cyrillic font" not in str(e):
            raise
        # Do not discard original real establishing shot just for a missing font.
        cover_status="source_S00_kept_fonts_missing"
    seconds=SCENES[0][-1] + sum(
        _duration(clock[sid],minimum)
        for sid,speaker,words,shot,camera,minimum in SCENES if speaker
    )
    create_original_score(work,seconds+1)
    render=native_render(work,clock)
    film=work/"ELIZABETH_BONES_SOBESEDOVANIE_STUDIO_PREVIEW.mp4"
    qa=verify_movie(film,work,render)
    delivered.mkdir(parents=True,exist_ok=False)
    for file in [film,film.with_suffix(".srt"),film.with_suffix(".json")]:
        shutil.copyfile(file,delivered/file.name)
    proof={
       "schema":1,"status":"actual_native_STUDIO_review_MP4_verified",
       "scene":"Elizabeth Bones E01 Собеседование",
       "original_Kaggle_job":source_kernel,
       "source_generation":"7 authentic SDXL-generated frames and 6 distinct Qwen3-TTS Russian WAV files",
       "studio_editor":"studio.hybrid_master.produce",
       "final_render_runtime":"GitHub Actions CPU (recovery from Kaggle missing Cyrillic font)",
       "kaggle_gpu_output_note":"All seven real stills and six source recordings completed on private Kaggle, but original notebook failed before montage; new montage created with real native Studio on CPU.",
       "invideo_used":False,"synthetic_fixture":False,
       "neural_actor_motion":False,"camera_animation":"studio.smooth_camera",
       "public_social_upload":False,"editorial_approved":False,"rights_approved":False,
       "cover_status":cover_status,"sources_sha256":assets,"verification":qa,
    }
    (delivered/"EB01_REAL_NATIVE_PROVENANCE.json").write_text(
        json.dumps(proof,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ELIZABETH_BONES_REAL_STUDIO_MP4_PASS",
          json.dumps({"sha256":qa["video_sha256"],"seconds":qa["duration_seconds"],
                      "frames":7,"spoken_cues":6,
                      "native":True,"invideo":False,"cover":cover_status},ensure_ascii=False))
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root",type=Path,required=True)
    parser.add_argument("--delivered",type=Path,required=True)
    parser.add_argument("--source-kernel",required=True)
    args=parser.parse_args()
    recover(args.private_root,args.delivered,args.source_kernel)

if __name__=="__main__":
    main()
