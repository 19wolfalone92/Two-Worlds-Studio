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

from PIL import Image, ImageDraw, ImageFont, ImageStat, ImageOps,ImageEnhance

VERSION = "1.0.0"
SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_AUDIO = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
ALLOWED_VIDEO = {".mp4", ".mov", ".webm"}


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


def video_duration(path: Path) -> float:
    info = probe(path)
    if not any(stream.get("codec_type") == "video" for stream in info.get("streams", [])):
        raise Blocked(f"Motion source has no video stream: {path.name}")
    length = float(info.get("format", {}).get("duration") or 0)
    if not math.isfinite(length) or length <= .1:
        raise Blocked(f"Invalid motion video length: {path.name}")
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
        motion = asset(base, scene["motion_video"], ALLOWED_VIDEO) if scene.get("motion_video") else None
        if motion and video_duration(motion) + .05 < length:
            raise Blocked(f'Scene {scene["id"]}: motion video shorter than Russian narration')
        takes=scene.get('motion_clips',[])
        if not isinstance(takes,list) or len(takes)>9:raise Blocked('Invalid bounded neural take list')
        takes=[asset(base,p,ALLOWED_VIDEO) for p in takes]
        for take in takes:video_duration(take)
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
                        "music": music, "motion_video": motion, "motion_clips":takes,"seconds": length, "cues": cues})
    return {"id": doc["id"], "title": title, "base": base, "script": script,
            "digest": digest, "width": width, "height": height,
            "fps": fps, "entries": entries, "smoke": smoke,
            "ai_disclosure":doc.get('ai_disclosure'),
            "master_audio": asset(base,doc['master_audio'],ALLOWED_AUDIO) if doc.get('master_audio') else None}


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


def write_srt(entries: list[dict], dest: Path, fps: int) -> None:
    lines = []
    offset = 0.0
    for scene in entries:
        for cue in scene["cues"]:
            lines += [str(len(lines) // 4 + 1),
                      f'{stamp(offset + cue["start"])} --> {stamp(offset + cue["end"])}',
                      cue["text"].strip().replace("-->", "→").replace("\n", " "), ""]
        offset += math.ceil(scene["seconds"] * fps) / fps
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ass(entries: list[dict], dest: Path, fps: int,disclosure: str | None = None) -> None:
    """Explicit portrait canvas and margins keep captions above TikTok controls."""
    header = ('[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n'
              'WrapStyle: 0\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n'
              'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, '
              'Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, '
              'Alignment, MarginL, MarginR, MarginV, Encoding\n'
              'Style: Narration,DejaVu Sans,54,&H00FFFFFF,&H00FFFFFF,&H19000000,&H70000000,'
              '-1,0,0,0,100,100,0,0,1,3,1,2,76,156,320,1\n\n[Events]\n'
              'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n')
    def ass_time(value):
        centis = round(value * 100)
        hours, centis = divmod(centis, 360000)
        minutes, centis = divmod(centis, 6000)
        secs, centis = divmod(centis, 100)
        return f'{hours}:{minutes:02}:{secs:02}.{centis:02}'
    lines, offset = [header], 0.0
    for scene in entries:
        for cue in scene['cues']:
            text = cue['text'].strip().replace('\\', '＼').replace('{', '(').replace('}', ')').replace('\n', ' ')
            lines.append(f"Dialogue: 0,{ass_time(offset+cue['start'])},{ass_time(offset+cue['end'])},Narration,,0,0,0,,{text}\n")
        offset += math.ceil(scene['seconds'] * fps)/fps
    if disclosure:
        text=disclosure.replace('\\','＼').replace('{','(').replace('}',')').replace('\n',' ')
        lines.append(f"Dialogue: 1,{ass_time(max(0,offset-4))},{ass_time(offset)},Narration,,0,0,0,,{{\\an8\\fs30\\bord1\\pos(500,115)}}{text}\n")
    dest.write_text(''.join(lines), encoding='utf-8')


def cover(image: Path, title: str, output: Path, w: int, h: int) -> None:
    with Image.open(image) as source:
        picture = ImageOps.fit(source.convert("RGB"), (w, h), method=Image.Resampling.LANCZOS)
    picture=ImageEnhance.Color(picture).enhance(1.10)
    picture=ImageEnhance.Brightness(picture).enhance(1.05)
    shade=Image.new('RGBA',(w,h),(0,0,0,0));sd=ImageDraw.Draw(shade)
    edge=round(h*.32)
    for y in range(edge):sd.line((0,y,w,y),fill=(2,7,15,round(145*(1-y/edge)**1.5)))
    picture=Image.alpha_composite(picture.convert('RGBA'),shade).convert('RGB')
    dr = ImageDraw.Draw(picture, "RGBA")
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


def render_scene(scene: dict, output: Path, w: int, h: int, fps: int,
                 subtitles_path: Path | None = None) -> None:
    if scene.get('motion_clips'):
        from .cinematic_edit import render
        receipt=render(scene,output,w,h,fps,subtitles_path)
        output.with_suffix('.edit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        return
    # Smooth photographic push-in, not an artificial skeleton/cutout animation.
    frames = math.ceil(scene["seconds"] * fps)
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if scene["motion_video"]:
        args += ["-i", str(scene["motion_video"])]
    else:
        from .smooth_camera import render_still
        camera_clip = output.with_suffix('.camera.mp4')
        render_still(scene['image'], camera_clip, w, h, fps, frames, 'push_in', strength=.35)
        args += ['-i', str(camera_clip)]
    args += ["-i", str(scene["voice"])]
    with_music = scene["music"] is not None
    if with_music:
        args += ["-stream_loop", "-1", "-i", str(scene["music"])]
    if scene["motion_video"]:
        motion_filter = (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
                         f"crop={w}:{h},fps={fps},format=yuv420p[v]")
    else:
        motion_filter = '[0:v]format=yuv420p[v]'
    if subtitles_path:
        escaped=str(subtitles_path).replace('\\','\\\\').replace(':','\\:').replace("'","\\'")
        motion_filter=motion_filter.replace('[v]',f",ass='{escaped}'[v]")
    filt = motion_filter + ";[1:a]aresample=48000,aformat=channel_layouts=stereo,apad[voice]"
    if with_music:
        filt += (";[voice]asplit=2[mainvoice][keyvoice];"
                 "[2:a]aresample=48000,aformat=channel_layouts=stereo,"
                 "volume=0.22[bed];"
                 "[bed][keyvoice]sidechaincompress=threshold=0.025:ratio=8:"
                 "attack=30:release=350[ducked];"
                 "[mainvoice][ducked]amix=inputs=2:duration=first:"
                 "normalize=0,alimiter=limit=0.92[a]")
    else:
        filt += ";[voice]alimiter=limit=0.92[a]"
    args += ["-filter_complex", filt, "-map", "[v]", "-map", "[a]",
             "-frames:v", str(frames), "-c:v", "libx264", "-preset", "medium",
             "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(fps),
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
             "-t", f"{frames/fps:.6f}", "-movflags", "+faststart", str(output)]
    try:
        _run(args)
    finally:
        if not scene['motion_video']:
            camera_clip.unlink(missing_ok=True)


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
              for p in (scene["image"], scene["voice"], scene["music"], scene["motion_video"]) if p}
    assets.update({str(p.relative_to(item['base'])):sha(p) for scene in item['entries'] for p in scene['motion_clips']})
    if item['master_audio']:
        length=duration(item['master_audio'])
        expected=sum(math.ceil(s['seconds']*item['fps'])/item['fps'] for s in item['entries'])
        if abs(length-expected)>.08:
            raise Blocked('Mixed story audio differs from the full original narration timeline')
        assets[str(item['master_audio'].relative_to(item['base']))]=sha(item['master_audio'])
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
        write_srt(item["entries"], staging / "subtitles.srt", item["fps"])
        write_ass(item['entries'], staging/'subtitles.ass', item['fps'],item['ai_disclosure'])
        for index, scene in enumerate(item["entries"], 1):
            scene_captions=clips/f'{index:03d}.ass'
            write_ass([scene],scene_captions,item['fps'],item['ai_disclosure'] if index==len(item['entries']) else None)
            render_scene(scene, clips / f"{index:03d}.mp4",
                         item["width"], item["height"], item["fps"],subtitles_path=scene_captions)
        playlist = staging / "clips.txt"
        playlist.write_text(
            "".join(f"file '{p.resolve()}'\n" for p in sorted(clips.glob("*.mp4"))),
            encoding="utf-8")
        _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
              "-i", str(playlist), "-c", "copy", str(staging / "joined.mp4")])
        # Captions were burned once per scene; the full portrait movie does
        # not need a second costly H.264 encode with the same subtitle pixels.
        args=['ffmpeg','-v','error','-nostdin','-y','-i',str(staging/'joined.mp4')]
        if item['master_audio']:
            args+=['-i',str(item['master_audio']),'-map','0:v:0','-map','1:a:0',
                   '-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-ac','2']
        else:
            args+=['-c','copy']
        args+=['-movflags','+faststart',str(staging/'final.mp4')]
        _run(args)
        cover(item["entries"][0]["image"], item["title"], staging / "cover.jpg",
              item["width"], item["height"])
        report = qa(staging / "final.mp4", item["width"], item["height"], item["fps"])
        report["smoke_only"] = smoke
        report["real_motion_video_scenes"] = sum(bool(s["motion_video"]) for s in item["entries"])
        report['neural_clip_montage_scenes']=sum(bool(s['motion_clips']) for s in item['entries'])
        report['editorial_shots']=[json.loads(p.read_text()) for p in sorted(clips.glob('*.edit.json'))]
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
