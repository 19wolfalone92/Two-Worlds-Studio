"""Fail-closed offline renderer for rights-reviewed Russian story media.

No neural generation, script writing, publishing, paid calls or automatic approval.
Requires real scene images, recorded voices and voice-aligned cue timestamps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat, ImageOps

VERSION = "1.0.0"
SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_AUDIO = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


class Blocked(ValueError):
    """Missing/invalid input must never be replaced with fake media."""


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def asset(base: Path, value: str, allowed: set[str]) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise Blocked(f"Unsafe or absent relative asset path: {value!r}")
    raw = Path(value)
    if ".." in raw.parts or raw.suffix.lower() not in allowed:
        raise Blocked(f"Unsafe or unsupported media path: {value!r}")
    path = (base / raw).resolve(strict=False)
    if base.resolve() not in path.parents or not path.is_file():
        raise Blocked(f"Missing or escaped media file: {value}")
    return path


def probe(path: Path) -> dict:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise Blocked(f"Cannot read media {path.name}: {p.stderr[:200]}")
    return json.loads(p.stdout)


def duration(path: Path) -> float:
    data = probe(path)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise Blocked(f"Audio stream missing: {path.name}")
    length = float(data.get("format", {}).get("duration") or 0)
    if not math.isfinite(length) or not .1 <= length <= 600:
        raise Blocked(f"Invalid voice duration: {path.name}: {length}")
    return length


def validate_project(manifest: Path, *, smoke: bool = False) -> dict:
    manifest = Path(manifest).resolve()
    base = manifest.parent
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    if (doc.get("schema") != 1 or not isinstance(doc.get("id"), str)
            or not SLUG.fullmatch(doc["id"])):
        raise Blocked("Invalid manifest schema or story id")
    if doc.get("language") != "ru":
        raise Blocked("Only explicit Russian story projects supported")
    title = doc.get("title")
    if not isinstance(title, str) or not title.strip():
        raise Blocked("Missing Russian story title")
    width, height, fps = (doc.get(k) for k in ("width", "height", "fps"))
    if not all(type(x) is int for x in (width, height, fps)):
        raise Blocked("Width/height/fps must be explicit integers")
    if smoke:
        if not (160 <= width <= 1080 and 280 <= height <= 1920 and 10 <= fps <= 30):
            raise Blocked("Invalid technical-smoke dimensions")
    elif (width, height, fps) not in ((1080, 1920, 24), (1080, 1920, 30)):
        raise Blocked("Master must be 1080x1920 and 24/30fps")
    if width % 2 or height % 2:
        raise Blocked("Dimensions must be even")
    script = asset(base, doc.get("script_file"), {".txt"})
    digest = doc.get("script_sha256")
    if not isinstance(digest, str) or digest != sha(script):
        raise Blocked("Script SHA-256 mismatch: existing story cannot be rewritten")
    scenes = doc.get("scenes")
    if not isinstance(scenes, list) or len(scenes) < (2 if smoke else 8):
        raise Blocked("Expected at least 8 real scenes (2 in technical smoke)")
    seen = set()
    entries = []
    for scene in scenes:
        if not isinstance(scene, dict) or type(scene.get("id")) is not int or scene["id"] in seen:
            raise Blocked("Invalid/duplicate scene id")
        seen.add(scene["id"])
        image = asset(base, scene.get("image"), ALLOWED_IMAGE)
        voice = asset(base, scene.get("voice"), ALLOWED_AUDIO)
        try:
            with Image.open(image) as im:
                im.verify()
        except Exception as exc:
            raise Blocked(f"Broken scene image {image.name}") from exc
        length = duration(voice)
        music = asset(base, scene["music"], ALLOWED_AUDIO) if scene.get("music") else None
        if music:
            duration(music)
        cues = scene.get("subtitles")
        if not isinstance(cues, list) or not cues:
            raise Blocked(f'Scene {scene["id"]} lacks real audio-aligned subtitles')
        previous = 0.0
        for cue in cues:
            if (not isinstance(cue, dict) or not isinstance(cue.get("text"), str)
                    or not cue["text"].strip()):
                raise Blocked("Invalid subtitle text")
            begin, end = cue.get("start"), cue.get("end")
            if (type(begin) not in (int, float) or type(end) not in (int, float)
                    or not math.isfinite(begin) or not math.isfinite(end)
                    or begin < previous - .001 or end <= begin + .05
                    or end > length + .06):
                raise Blocked(f'Invalid real subtitle timestamp in scene {scene["id"]}')
            previous = end
        entries.append({"id": scene["id"], "image": image, "voice": voice,
                        "music": music, "seconds": length, "cues": cues})
    return {"id": doc["id"], "title": title, "base": base, "script": script,
            "digest": digest, "width": width, "height": height,
            "fps": fps, "entries": entries, "smoke": smoke}


def _run(args: list[str]) -> None:
    p = subprocess.run(args, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"FFmpeg failed ({p.returncode}): {p.stderr[-1700:]}")


def stamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(entries: list[dict], dest: Path) -> None:
    lines = []
    offset = 0.0
    for scene in entries:
        for cue in scene["cues"]:
            lines += [str(len(lines) // 4 + 1),
                      f'{stamp(offset + cue["start"])} --> {stamp(offset + cue["end"])}',
                      cue["text"].strip().replace("-->", "→").replace("\n", " "), ""]
        offset += scene["seconds"]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cover(image: Path, title: str, output: Path, w: int, h: int) -> None:
    with Image.open(image) as source:
        picture = ImageOps.fit(source.convert("RGB"), (w, h), method=Image.Resampling.LANCZOS)
    dr = ImageDraw.Draw(picture, "RGBA")
    dr.rectangle((0, 0, w, int(h * .28)), fill=(2, 7, 15, 125))
    font_size = max(20, round(w * .063))
    fonts = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"]
    font = next((ImageFont.truetype(p, font_size) for p in fonts if Path(p).is_file()),
                ImageFont.load_default())
    words, rows, line = title.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if dr.textbbox((0, 0), trial, font=font)[2] > w * .86 and line:
            rows.append(line)
            line = word
        else:
            line = trial
    rows.append(line)
    for index, text in enumerate(rows[:3]):
        x, y = round(w * .06), round(h * .07) + index * round(font_size * 1.2)
        dr.text((x, y), text, font=font, fill="white", stroke_width=3,
                stroke_fill=(0, 0, 0, 210))
    picture.save(output, quality=94, subsampling=0)


def render_scene(scene: dict, output: Path, w: int, h: int, fps: int) -> None:
    # Smooth photographic push-in, not an artificial skeleton/cutout animation.
    frames = math.ceil(scene["seconds"] * fps)
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-loop", "1", "-framerate", str(fps), "-i", str(scene["image"]),
            "-i", str(scene["voice"])]
    with_music = scene["music"] is not None
    if with_music:
        args += ["-stream_loop", "-1", "-i", str(scene["music"])]
    scale_w, scale_h = int(w * 1.17) // 2 * 2, int(h * 1.17) // 2 * 2
    filt = (
        f"[0:v]scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase,"
        f"crop={scale_w}:{scale_h},"
        f"zoompan=z='min(zoom+0.0007,1.09)':x='(iw-iw/zoom)/2':"
        f"y='(ih-ih/zoom)/2':d=1:s={w}x{h}:fps={fps},format=yuv420p[v];"
        "[1:a]aresample=48000,aformat=channel_layouts=stereo,apad[voice]"
    )
    if with_music:
        filt += (";[2:a]aresample=48000,aformat=channel_layouts=stereo,"
                 "volume=0.13[bed];[voice][bed]amix=inputs=2:duration=first:"
                 "normalize=0,alimiter=limit=0.92[a]")
    else:
        filt += ";[voice]alimiter=limit=0.92[a]"
    args += ["-filter_complex", filt, "-map", "[v]", "-map", "[a]",
             "-frames:v", str(frames), "-c:v", "libx264", "-preset", "medium",
             "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(fps),
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
             "-t", f"{frames/fps:.6f}", "-movflags", "+faststart", str(output)]
    _run(args)


def qa(video: Path, w: int, h: int, fps: int) -> dict:
    info = probe(video)
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    correct = (v is not None and a is not None and v.get("codec_name") == "h264"
               and a.get("codec_name") == "aac" and v.get("width") == w
               and v.get("height") == h and v.get("pix_fmt") == "yuv420p")
    if not correct:
        raise RuntimeError("Final stream codec or size mismatch")
    f = v.get("avg_frame_rate", "0/1").split("/")
    if len(f) != 2 or not int(f[1]) or abs(int(f[0]) / int(f[1]) - fps) > .01:
        raise RuntimeError("Wrong output frame rate")
    _run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"])
    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp) / "first.png"
        _run(["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", "1", str(frame)])
        with Image.open(frame) as im:
            mean = sum(ImageStat.Stat(im.convert("RGB")).mean) / 3
        if mean < 9:
            raise RuntimeError(f"First frame is near-black: brightness={mean:.1f}")
    return {"technical_pass": True, "full_decode_pass": True,
            "first_frame_mean": round(mean, 2),
            "width": w, "height": h, "fps": fps,
            "duration_s": float(info.get("format", {}).get("duration", 0)),
            "human_artistic_review": "pending", "rights_review": "pending",
            "approved_for_publication": False}


def run(manifest: Path, destination: Path, *, smoke: bool = False,
        dry_run: bool = False) -> dict:
    item = validate_project(manifest, smoke=smoke)
    target = Path(destination).resolve() / item["id"]
    if target.exists():
        raise Blocked(f"Output exists; refuse silent overwrite: {target}")
    assets = {str(p.relative_to(item["base"])): sha(p)
              for scene in item["entries"]
              for p in (scene["image"], scene["voice"], scene["music"]) if p}
    if dry_run:
        return {"status": "ASSETS_READY", "dry_run": True,
                "story": item["id"], "scenes": len(item["entries"]),
                "source_sha256": item["digest"], "assets": assets}
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".shorts-build-", dir=destination) as td:
        staging = Path(td)
        clips = staging / "clips"
        clips.mkdir()
        for index, scene in enumerate(item["entries"], 1):
            render_scene(scene, clips / f"{index:03d}.mp4",
                         item["width"], item["height"], item["fps"])
        playlist = staging / "clips.txt"
        playlist.write_text(
            "".join(f"file '{p.resolve()}'\n" for p in sorted(clips.glob("*.mp4"))),
            encoding="utf-8")
        _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
              "-i", str(playlist), "-c", "copy", str(staging / "joined.mp4")])
        write_srt(item["entries"], staging / "subtitles.srt")
        srt = (str(staging / "subtitles.srt").replace("\\", "\\\\")
               .replace(":", "\\:").replace("'", "\\'"))
        style = "FontName=DejaVu Sans,FontSize=17,Outline=2,Shadow=0,Alignment=2,MarginV=280"
        _run(["ffmpeg", "-v", "error", "-nostdin", "-y",
              "-i", str(staging / "joined.mp4"),
              "-vf", f"subtitles='{srt}':force_style='{style}'",
              "-c:v", "libx264", "-preset", "medium", "-crf", "18",
              "-pix_fmt", "yuv420p", "-c:a", "copy",
              "-movflags", "+faststart", str(staging / "final.mp4")])
        cover(item["entries"][0]["image"], item["title"], staging / "cover.jpg",
              item["width"], item["height"])
        report = qa(staging / "final.mp4", item["width"], item["height"], item["fps"])
        report["smoke_only"] = smoke
        report["music_for_all_scenes"] = all(s["music"] for s in item["entries"])
        (staging / "qa_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        receipt = {"schema": 1, "engine": VERSION, "story_id": item["id"],
                   "script_sha256": item["digest"], "assets_sha256": assets,
                   "final_sha256": sha(staging / "final.mp4"),
                   "review_status": "REVIEW_REQUIRED",
                   "approved_for_publication": False}
        (staging / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.rmtree(clips)
        (staging / "joined.mp4").unlink()
        playlist.unlink()
        os.rename(staging, target)
    return {"status": "REVIEW_REQUIRED", "story": item["id"],
            "files": [str(target / fn) for fn in (
                "final.mp4", "cover.jpg", "subtitles.srt", "qa_report.json", "receipt.json")],
            "qa": report}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Two Worlds Russian story shorts — real media only")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--technical-smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(args.project, args.out, smoke=args.technical_smoke,
                     dry_run=args.dry_run)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
