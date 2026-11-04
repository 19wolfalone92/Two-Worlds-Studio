"""Offline hybrid short-video editor: stills + optional pre-generated motion clips.

No cloud service, model download, automatic publishing or fabricated approvals.
JSON shot list is authored externally (e.g. approved by Work). FFmpeg does
camera motion, stitches shots, aligns dialogue, mixes optional soundtrack,
and exports one MP4 plus separate SRT and a reproducible QA receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from studio.smooth_camera import CAMERAS, render_still
IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
CLIPS = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def source(base: Path, rel: str, suffixes: set[str]) -> Path:
    """Explicitly relative, in-project sources; prohibit path escapes/symlinks."""
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        raise ValueError("Media paths must be nonempty and relative to the manifest")
    root = base.resolve()
    path = (root / rel).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"Missing or out-of-project asset: {rel}")
    if path.suffix.lower() not in suffixes:
        raise ValueError(f"Unsupported media type: {rel}")
    return path


def probe(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=30)
    return json.loads(proc.stdout)


def has_stream(path: Path, kind: str) -> bool:
    return any(s.get("codec_type") == kind for s in probe(path).get("streams", []))


def check_manifest(data: dict, base: Path, *, master: bool = False) -> dict:
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ValueError("Expected hybrid shot manifest with schema=1")
    width, height, fps = data.get("width", 1080), data.get("height", 1920), data.get("fps", 24)
    if not all(type(i) is int for i in (width, height, fps)):
        raise ValueError("Resolution and fps must be integers")
    if not (width >= 288 and height >= 512 and width <= 2160 and height <= 3840
            and width % 2 == 0 and height % 2 == 0 and abs(width / height - 9 / 16) < .001
            and 12 <= fps <= 30):
        raise ValueError("Expected an even 9:16 resolution and 12-30 fps")
    shots = data.get("shots")
    if not isinstance(shots, list) or not (1 <= len(shots) <= 80):
        raise ValueError("Provide 1-80 shots")
    if master and not (data.get("editorial_approved") is True and
                       data.get("rights_approved") is True):
        raise ValueError("Master requires explicit editorial and rights approvals")
    normalized, start = [], 0
    for i, item in enumerate(shots):
        if not isinstance(item, dict):
            raise ValueError(f"Shot {i} is not an object")
        duration = item.get("duration")
        if type(duration) not in (float, int) or not math.isfinite(duration) or not 0.5 <= duration <= 30:
            raise ValueError(f"Shot {i} duration must be 0.5-30 seconds")
        frames = round(duration * fps)
        if frames < 1:
            raise ValueError(f"Shot {i} is too short")
        kind = "clip" if "clip" in item else "image" if "image" in item else None
        if kind is None or ("clip" in item and "image" in item):
            raise ValueError(f"Shot {i} needs exactly one image or clip")
        media = source(base, item[kind], CLIPS if kind == "clip" else IMAGES)
        if kind == "clip" and not has_stream(media, "video"):
            raise ValueError(f"Shot {i}: clip has no video stream")
        camera = item.get("camera", "push_in")
        if camera not in CAMERAS:
            raise ValueError(f"Shot {i}: unknown camera move {camera}")
        strength = item.get("camera_strength", 0.45)
        if type(strength) not in (float, int) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError(f"Shot {i}: camera_strength must be between 0 and 1")
        caption = item.get("caption", "")
        if not isinstance(caption, str) or len(caption) > 240:
            raise ValueError(f"Shot {i}: caption too long")
        approved = item.get("approved") is True
        if master and not approved:
            raise ValueError(f"Shot {i}: asset not explicitly approved")
        voice = None
        if item.get("voice"):
            voice = source(base, item["voice"], AUDIO)
            if not has_stream(voice, "audio"):
                raise ValueError(f"Shot {i}: no audio in voice track")
            voice_duration = float(probe(voice).get("format", {}).get("duration", 0))
            if voice_duration > frames / fps + .15:
                raise ValueError(f"Shot {i}: voice exceeds shot duration; extend shot or trim speech")
            if master and item.get("voice_approved") is not True:
                raise ValueError(f"Shot {i}: voice must be explicitly approved")
        normalized.append(dict(kind=kind, path=media, frames=frames,
                               duration=frames / fps, start=start / fps, camera=camera, strength=strength,
                               caption=caption.strip(), voice=voice, approved=approved))
        start += frames
    if start / fps > 600:
        raise ValueError("Video longer than 10 minutes")
    music = source(base, data["music"], AUDIO) if data.get("music") else None
    if music is not None and not has_stream(music, "audio"):
        raise ValueError("Music source has no audio")
    if master and music and data.get("music_approved") is not True:
        raise ValueError("Master soundtrack must be explicitly approved")
    volume = data.get("music_volume", .16)
    if type(volume) not in (int, float) or not math.isfinite(volume) or not 0 <= volume <= .5:
        raise ValueError("music_volume must be between 0 and 0.5")
    return dict(width=width, height=height, fps=fps, shots=normalized,
                frames=start, duration=start / fps, music=music, music_volume=volume,
                approved=master)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, timeout=1200)


def render_segment(shot: dict, path: Path, width: int, height: int, fps: int) -> None:
    if shot["kind"] == "image":
        render_still(shot["path"], path, width, height, fps,
                     shot["frames"], shot["camera"], shot["strength"])
        return
    inputs = ["-i", str(shot["path"])]
    # Existing AI clips are normalized to target frame rate, hold last frame
    # if short; no invented movement or interpolation of subject identity.
    filters = (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
               f"crop={width}:{height},setsar=1,fps={fps},"
               f"tpad=stop_mode=clone:stop_duration={shot['duration']},format=yuv420p")
    run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         *inputs, "-vf", filters, "-frames:v", str(shot["frames"]),
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
         "-pix_fmt", "yuv420p", "-r", str(fps), str(path)])


def timestamp(t: float) -> str:
    ms = round(t * 1000)
    h, r = divmod(ms, 3600000)
    m, r = divmod(r, 60000)
    s, ms = divmod(r, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def subtitles(shots: list[dict]) -> str:
    lines = []
    for shot in shots:
        if shot["caption"]:
            n = len(lines) + 1
            lines.append(f"{n}\n{timestamp(shot['start'])} --> "
                         f"{timestamp(shot['start'] + shot['duration'])}\n{shot['caption']}\n")
    return "\n".join(lines) + ("\n" if lines else "")


def add_audio(video: Path, output: Path, plan: dict) -> None:
    inputs = ["-i", str(video)]
    audio_inputs = []
    if plan["music"]:
        inputs += ["-stream_loop", "-1", "-i", str(plan["music"])]
        audio_inputs.append((len(audio_inputs) + 1, "music", None))
    for shot in plan["shots"]:
        if shot["voice"]:
            inputs += ["-i", str(shot["voice"])]
            audio_inputs.append((len(audio_inputs) + 1, "voice", shot))
    if not audio_inputs:
        inputs += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        graph = f"[1:a]atrim=duration={plan['duration']},asetpts=PTS-STARTPTS[aout]"
    else:
        labels, steps = [], []
        for idx, kind, shot in audio_inputs:
            label = f"s{idx}"
            if kind == "music":
                steps.append(f"[{idx}:a]aresample=48000,volume={plan['music_volume']},"
                             f"atrim=duration={plan['duration']},asetpts=PTS-STARTPTS[{label}]")
            else:
                delay = round(shot["start"] * 1000)
                steps.append(f"[{idx}:a]aresample=48000,"
                             f"atrim=duration={shot['duration']},adelay={delay}:all=1,"
                             f"volume=1.0[{label}]")
            labels.append(f"[{label}]")
        steps.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:"
                     f"normalize=0,atrim=duration={plan['duration']},"
                     "alimiter=limit=0.90[aout]")
        graph = ";".join(steps)
    run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         *inputs, "-filter_complex", graph, "-map", "0:v:0", "-map", "[aout]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
         "-t", f"{plan['duration']:.5f}", "-movflags", "+faststart", str(output)])


def produce(manifest: Path, output: Path, *, master: bool = False) -> dict:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg and ffprobe must be installed")
    manifest = manifest.resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    plan = check_manifest(data, manifest.parent, master=master)
    output = output.resolve()
    if output.suffix.lower() != ".mp4" or output == manifest:
        raise ValueError("Output must be a new .mp4 file")
    if any(output == s["path"] for s in plan["shots"]):
        raise ValueError("Output would overwrite a source asset")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tw-hybrid-") as td:
        temp = Path(td)
        segments = []
        for n, shot in enumerate(plan["shots"]):
            segment = temp / f"shot_{n:03d}.mp4"
            render_segment(shot, segment, plan["width"], plan["height"], plan["fps"])
            segments.append(segment)
        concat = temp / "concat.txt"
        concat.write_text("".join(f"file '{p.name}'\n" for p in segments), encoding="utf-8")
        silent = temp / "silent.mp4"
        run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "concat", "-safe", "0", "-i", str(concat),
             "-c:v", "copy", str(silent)])
        staged = temp / "mixed.mp4"
        add_audio(silent, staged, plan)
        measured = probe(staged)
        streams = measured.get("streams", [])
        videos = [s for s in streams if s.get("codec_type") == "video"]
        audios = [s for s in streams if s.get("codec_type") == "audio"]
        seconds = float(measured.get("format", {}).get("duration", 0))
        if (len(videos) != 1 or len(audios) != 1 or
                videos[0].get("width") != plan["width"] or
                videos[0].get("height") != plan["height"] or
                abs(seconds - plan["duration"]) > .2):
            raise RuntimeError("Export failed video/audio/duration quality checks")
        shutil.copyfile(staged, output)
    output.with_suffix(".srt").write_text(subtitles(plan["shots"]), encoding="utf-8")
    receipt = {
        "schema": 1, "source_manifest": manifest.name,
        "source_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "resolution": [plan["width"], plan["height"]], "fps": plan["fps"],
        "duration_seconds": round(seconds, 3), "shots": len(plan["shots"]),
        "motion_clips": sum(s["kind"] == "clip" for s in plan["shots"]),
        "static_camera_shots": sum(s["kind"] == "image" for s in plan["shots"]),
        "audio_tracks": len([s for s in plan["shots"] if s["voice"]]) + bool(plan["music"]),
        "review_status": "asset_approved_not_published" if master else "preview_unreviewed",
        "published": False,
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description="Offline hybrid short-video studio, never posts to platforms")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--master", action="store_true", help="Require explicit per-asset approvals")
    a = p.parse_args()
    print(json.dumps(produce(a.manifest, a.output, master=a.master), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
