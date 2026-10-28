"""Review-gated, deterministic production and publication package builder.

This program renders from approved *prewritten* episode specs. It never uses AI APIs,
contacts TikTok/Metricool, or publishes on its own.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from studio.render import render, validate_episode
from studio.continuity import check_episode
from studio.premium_assets import review_episode

ROOT = Path(__file__).resolve().parent.parent
EPISODE_ID = re.compile(r"^E[0-9]{2,4}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def safe_episode_path(rel: str) -> Path:
    if not isinstance(rel, str) or not rel.startswith("episodes/"):
        raise ValueError("Episode must be under episodes/")
    p = (ROOT / rel).resolve()
    if (ROOT / "episodes").resolve() not in p.parents or p.suffix != ".json":
        raise ValueError("Invalid episode path")
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def scheduled_episodes(plan_path: Path, on_date: dt.date) -> list[Path]:
    plan = read_json(plan_path)
    if not isinstance(plan.get("episodes"), list):
        raise ValueError("release_plan.json requires an episodes array")
    found = []
    seen = set()
    for item in plan["episodes"]:
        if not isinstance(item, dict) or not EPISODE_ID.fullmatch(str(item.get("id", ""))):
            raise ValueError("Invalid release plan episode ID")
        if item["id"] in seen:
            raise ValueError("Duplicate release plan episode ID")
        seen.add(item["id"])
        date_s = item.get("release_date")
        if date_s is None:
            continue
        if not isinstance(date_s, str) or not DATE.fullmatch(date_s):
            raise ValueError("release_date must be an ISO UTC day or null")
        release_date = dt.date.fromisoformat(date_s)
        if release_date != on_date:
            continue
        if not (item.get("editorial_approved") is True and item.get("rights_approved") is True):
            raise ValueError(f"{item['id']} scheduled without both approvals")
        path = safe_episode_path(item.get("episode_file", ""))
        episode = read_json(path)
        if episode.get("id") != item["id"]:
            raise ValueError("Release plan ID does not match episode JSON")
        check_episode(episode, strict=True)
        # Scheduled release cannot silently use simple placeholder drawings.
        review_episode(episode,ROOT/"assets",publish=True)
        found.append(path)
    return found


def srt_timestamp(seconds: float) -> str:
    millis = round(seconds * 1000)
    h, remainder = divmod(millis, 3600000)
    m, remainder = divmod(remainder, 60000)
    s, ms = divmod(remainder, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def subtitles(scenes: list[dict], fps: int) -> str:
    t_frames = 0
    chunks = []
    for scene in scenes:
        frames = max(1, round(float(scene["duration"]) * fps))
        line = scene.get("caption", "").strip()
        if line:
            start = t_frames / fps
            stop = (t_frames + frames) / fps
            chunks.append(f"{len(chunks)+1}\n{srt_timestamp(start)} --> {srt_timestamp(stop)}\n{line}\n")
        t_frames += frames
    return "\n".join(chunks)


def inspect_mp4(path: Path, width: int, height: int, seconds: float) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_name,codec_type,width,height",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise RuntimeError("Preview must contain one video stream and one audio stream")
    if (videos[0].get("width"), videos[0].get("height")) != (width, height):
        raise RuntimeError("Preview dimensions do not match requested 9:16 resolution")
    if abs(width/height - 9/16) > 0.005:
        raise RuntimeError("TikTok video needs 9:16 ratio")
    duration = float(data["format"]["duration"])
    if abs(duration - seconds) > 0.35:
        raise RuntimeError(f"Preview duration mismatch: {duration:.2f} vs {seconds:.2f}")
    if path.stat().st_size < 10000:
        raise RuntimeError("Preview is unexpectedly small")
    return {"duration_seconds": round(duration,3), "resolution": f"{width}x{height}",
            "video_codec": videos[0].get("codec_name"),
            "audio_codec": audios[0].get("codec_name")}


def add_offline_voiceover(video: Path, scenes: list[dict], fps: int) -> None:
    """Optional English robotic TTS; adds speech to the existing ambient track.

    Uses system espeak-ng, not external voice/cloud APIs. Unnatural voices are
    expected in this technical preview. A licensed human/AI voice can replace it.
    """
    if not shutil.which("espeak-ng"):
        raise RuntimeError("Offline voiceover requested but espeak-ng is unavailable")
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        wavs = []
        at = 0
        for idx, scene in enumerate(scenes):
            frames = max(1,round(scene["duration"]*fps))
            speech = str(scene.get("voiceover") or "").strip()
            if speech:
                wav = temp_path / f"scene_{idx}.wav"
                subprocess.run(["espeak-ng", "-v", "en-us", "-s", "175", "-w", str(wav), speech],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                wavs.append((wav, round(at/fps*1000)))
            at += frames
        if not wavs:
            return
        # Filter graph delays spoken lines by the timeline position, mixes with soft bed.
        # Duration=first ensures no accidental extension beyond the comic.
        inputs = ["-i", str(video)]
        graph = ["[0:a]volume=0.35[bed]"]
        streams = ["[bed]"]
        for i, (wav, delay) in enumerate(wavs, start=1):
            inputs += ["-i", str(wav)]
            graph.append(f"[{i}:a]adelay={delay}:all=1,volume=1.0[v{i}]")
            streams.append(f"[v{i}]")
        graph.append("".join(streams) + f"amix=inputs={len(streams)}:duration=first:dropout_transition=0[aout]")
        tmp = temp_path / "with_voice.mp4"
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", *inputs, "-filter_complex",
               ";".join(graph), "-map", "0:v", "-map", "[aout]",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags",
               "+faststart", str(tmp)]
        subprocess.run(cmd,check=True)
        shutil.copyfile(tmp,video)


def produce_episode(path: Path, outdir: Path, width=540, height=960, fps=12, voiceover=False) -> dict:
    episode = read_json(path)
    eid = episode.get("id")
    if not isinstance(eid, str) or not EPISODE_ID.fullmatch(eid):
        raise ValueError("Invalid episode ID")
    scenes = validate_episode(episode, ROOT / "assets")
    continuity_audit = check_episode(episode)
    visual_audit = review_episode(episode,ROOT/"assets",publish=False,
                                 min_width=width,min_height=height)
    for scene in scenes:
        if len(scene.get("heading", "")) > 48 or len(scene.get("caption", "")) > 140:
            raise ValueError(f"{scene['id']}: text is too long for a small mobile panel")
        if scene.get("voiceover") is not None and len(scene["voiceover"]) > 160:
            raise ValueError(f"{scene['id']}: spoken line too long")
    target = outdir / eid
    target.mkdir(parents=True, exist_ok=True)
    video = target / f"{eid}_preview.mp4"
    info = render(path, video, ROOT/"assets", width, height, fps)
    if voiceover:
        add_offline_voiceover(video, scenes, fps)
    qc = inspect_mp4(video,width,height,info["duration_seconds"])
    (target/f"{eid}.srt").write_text(subtitles(scenes,fps),encoding="utf-8")
    caption = (f"TWO WORLDS — {eid}: {episode.get('title','THE STORY')}\n"
               "Two lives. Two worlds. One impossible exchange.\n"
               "Watch the next chapter. #TwoWorlds #MotionComic #DarkFantasy #ParallelWorlds")
    (target/"POSTING_PACKAGE.md").write_text(
        f"# TikTok / Shorts posting package: {eid}\n\n"
        f"**Status:** PREVIEW ONLY — HUMAN APPROVAL REQUIRED\n\n"
        f"Video: {video.name}\n\n"
        f"Caption:\n\n{caption}\n\n"
        f"Subtitle file: {eid}.srt\n\n"
        "Checklist: verify story continuity, human review of graphics and audio, "
        "rights clearance, AI-generated content labeling, TikTok account settings. "
        "The workflow does not upload or schedule anything.\n", encoding="utf-8"
    )
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    manifest = {
        "episode": eid, "title": episode.get("title"),
        "video": video.name, "sha256": digest, "subtitles":f"{eid}.srt",
        "publication_status": "awaiting_human_review",
        "is_ai_generated_label_recommended": True,
        "voiceover": "offline_espeak" if voiceover else "none",
        "artwork_mode": "licensed_user_assets_or_procedural_standin",
        "qc": qc, "continuity_audit": continuity_audit,
        "visual_audit": visual_audit
    }
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Offline motion comic production (never publishes)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode",help="Exact episode JSON path relative to repository, e.g. episodes/E01_THE_REFLECTION.json")
    group.add_argument("--scheduled",action="store_true",help="Prepare episodes explicitly approved for specified UTC date")
    parser.add_argument("--date",default=dt.datetime.now(dt.timezone.utc).date().isoformat(),help="UTC schedule date YYYY-MM-DD")
    parser.add_argument("--plan",type=Path,default=ROOT/"release_plan.json")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"dist")
    parser.add_argument("--width",type=int,default=540)
    parser.add_argument("--height",type=int,default=960)
    parser.add_argument("--fps",type=int,default=12)
    parser.add_argument("--voiceover",action="store_true",help="Use local espeak-ng robotic English voice")
    args=parser.parse_args()
    if not DATE.fullmatch(args.date):
        parser.error("--date must be YYYY-MM-DD")
    date=dt.date.fromisoformat(args.date)
    if args.scheduled:
        episodes=scheduled_episodes(args.plan,date)
        if not episodes:
            print(json.dumps({"date":str(date),"staged":0,"reason":"no approved episodes due"}))
            return
    else:
        episodes=[safe_episode_path(args.episode)]
    for episode in episodes:
        print(json.dumps(produce_episode(episode,args.output_dir,args.width,args.height,args.fps,args.voiceover)))
    print(f"Staged {len(episodes)} episode(s); no publication attempted.")


if __name__ == "__main__":
    main()
