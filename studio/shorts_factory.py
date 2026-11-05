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

from PIL import Image, ImageDraw, ImageStat, ImageOps,ImageEnhance

VERSION = "1.4.0"
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


def asset_hashes(item: dict) -> dict:
    paths = [p for scene in item['entries'] for p in (
        scene['image'], scene['voice'], scene['music'], scene['motion_video'],
        *scene['motion_clips']) if p]
    if item['master_audio']:
        paths.append(item['master_audio'])
    return {p.relative_to(item['base']).as_posix(): sha(p) for p in paths}


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
    manifest_bytes = manifest.read_bytes()
    doc = json.loads(manifest_bytes)
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
            with Image.open(image) as im:
                source_size = ImageOps.exif_transpose(im).size
        except Exception as exc:
            raise Blocked(f"Broken scene image {image.name}") from exc
        source_scale = max(width / source_size[0], height / source_size[1])
        if not smoke and source_scale > 2.0:
            raise Blocked(f"Scene {scene['id']}: source is a thumbnail; canvas-fill upscale exceeds 2x")
        from .smooth_camera import camera_pose, validate_framing
        try:
            camera_pose(0, 2, scene.get('camera', 'push_in'), scene.get('camera_strength', .35))
            framing = validate_framing(scene.get('framing'))
        except (ValueError, TypeError) as exc:
            raise Blocked(f"Scene {scene['id']}: invalid camera/framing: {exc}") from exc
        if framing and (scene.get('motion_video') or scene.get('motion_clips')):
            raise Blocked('Source focus applies to photographic scenes; author motion-clip crops separately')
        caption_position = scene.get('caption_position', 'bottom')
        if caption_position not in ('bottom', 'upper'):
            raise Blocked('Caption position must be bottom or upper')
        length = duration(voice)
        motion = asset(base, scene["motion_video"], ALLOWED_VIDEO) if scene.get("motion_video") else None
        if motion and video_duration(motion) + .05 < length:
            raise Blocked(f'Scene {scene["id"]}: motion video shorter than Russian narration')
        takes=scene.get('motion_clips',[])
        if not isinstance(takes,list) or len(takes)>9:raise Blocked('Invalid bounded neural take list')
        takes=[asset(base,p,ALLOWED_VIDEO) for p in takes]
        for take in takes:video_duration(take)
        focus=scene.get('foreground_soft_focus')
        if focus is not None and (not isinstance(focus,dict) or
            not all(isinstance(focus.get(k),(int,float)) and math.isfinite(focus[k]) for k in ['start','end','sigma'])
            or not 0<=focus['start']<focus['end']<=1 or not 1<=focus['sigma']<=30):
            raise Blocked('Invalid bounded foreground focus configuration')
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
        source_text = scene.get('source_text')
        if source_text is not None:
            if (not isinstance(source_text, str) or not source_text.strip()
                    or ' '.join(source_text.split()) !=
                    ' '.join(' '.join(c['text'] for c in cues).split())):
                raise Blocked(f"Scene {scene['id']}: captions changed or omitted source words")
        from .caption_layout import audit_cues
        try:
            caption_audit = audit_cues(cues)
        except ValueError as exc:
            raise Blocked(f"Scene {scene['id']}: {exc}") from exc
        entries.append({"id": scene["id"], "image": image, "voice": voice,
                        "music": music, "motion_video": motion, "motion_clips":takes,"seconds": length,
                        "foreground_soft_focus":focus,"cues": cues,
                        "camera":scene.get("camera","push_in"),
                        "camera_strength":scene.get("camera_strength",.35),
                        "framing":framing,"caption_position":caption_position,
                        "source_upscale":round(source_scale, 4),"caption_audit":caption_audit,
                        "caption_source_verified":source_text is not None})
    cover_scene = doc.get('cover_scene_id', entries[0]['id'])
    cover_text = doc.get('cover_text', title)
    if type(cover_scene) is not int or cover_scene not in seen:
        raise Blocked('Cover scene must name an existing scene id')
    if not isinstance(cover_text, str) or len(cover_text) > 120:
        raise Blocked('Cover text must be an authored string of at most 120 characters')
    return {"id": doc["id"], "title": title, "base": base, "script": script,
            'project_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
            "digest": digest, "width": width, "height": height,
            "fps": fps, "entries": entries, "smoke": smoke,
            "ai_disclosure":doc.get('ai_disclosure'), "intro_hook":doc.get('intro_hook'),
            "cover_scene_id":cover_scene,"cover_text":cover_text,
            "master_audio": asset(base,doc['master_audio'],ALLOWED_AUDIO) if doc.get('master_audio') else None}


def _run(args: list[str]) -> None:
    p = subprocess.run(args, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"FFmpeg failed ({p.returncode}): {p.stderr[-1700:]}")
    # Make the completed artifact durable before a following probe/transfer.
    # Intermediates and final MP4s must never be exposed with pending data.
    output=Path(args[-1])
    if output.is_file():
        with output.open('rb') as stream:os.fsync(stream.fileno())


def stamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def frame_count(seconds: float, fps: int) -> int:
    """Ceil real durations without adding a frame for ffprobe microsecond rounding."""
    return max(1, math.ceil(seconds * fps - .0001))


def write_srt(entries: list[dict], dest: Path, fps: int) -> None:
    from .caption_layout import layout
    lines = []
    offset = 0.0
    for scene in entries:
        for cue in scene["cues"]:
            lines += [str(len(lines) // 4 + 1),
                      f'{stamp(offset + cue["start"])} --> {stamp(offset + cue["end"])}',
                      '\n'.join(layout(cue['text'])['lines']).replace("-->", "→"), ""]
        offset += frame_count(scene["seconds"], fps) / fps
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ass(entries: list[dict], dest: Path, fps: int,disclosure: str | None = None,hook_text: str | None = None) -> None:
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
    from .caption_layout import ass_text
    for scene in entries:
        for cue in scene['cues']:
            text = ass_text(cue['text'])
            if scene.get('caption_position') == 'upper':
                text = r'{\an8\pos(500,380)}' + text
            lines.append(f"Dialogue: 0,{ass_time(offset+cue['start'])},{ass_time(offset+cue['end'])},Narration,,0,0,0,,{text}\n")
        offset += frame_count(scene['seconds'], fps)/fps
    if hook_text:
        text=ass_text(hook_text,preferred_size=64)
        lines.append(f"Dialogue: 2,0:00:00.00,{ass_time(min(2.0,offset))},Narration,,0,0,0,,{{\\an8\\bord3\\pos(500,185)}}{text}\n")
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
    from .caption_layout import layout,font as caption_font
    plan = (layout(title, preferred_size=font_size, minimum_size=max(16,round(w*.04)),
                   max_width=round(w*.86)) if title.strip() else
            {'lines':[],'font_size':font_size})
    font_size=plan['font_size'];font=caption_font(font_size)
    for index, text in enumerate(plan['lines']):
        x, y = round(w * .06), round(h * .07) + index * round(font_size * 1.2)
        dr.text((x, y), text, font=font, fill="white", stroke_width=3,
                stroke_fill=(0, 0, 0, 210))
    picture.save(output, quality=94, subsampling=0)


def editorial_report(entries: list[dict], fps: int) -> dict:
    """Expose pacing tradeoffs without inventing events or approving a story."""
    lengths = [frame_count(s['seconds'], fps)/fps for s in entries]
    seconds = sum(lengths)
    notes = []
    if seconds < 40:
        notes.append('Below the 40-second story target; check that the ending is complete')
    elif seconds > 75:
        notes.append('Above the 75-second ceiling; adapt the script, never accelerate the narrator')
    elif seconds > 65:
        notes.append('65-75 seconds: retain only if shorter writing would break the story')
    if not 8 <= len(entries) <= 12:
        notes.append('Outside the 8-12-scene guide; check whether each scene carries new information')
    long_holds = [s['id'] for s, length in zip(entries, lengths)
                  if length > 9 and not (s.get('motion_clips') or s.get('motion_video'))]
    if long_holds:
        notes.append('Photographic scenes longer than 9 seconds need editorial review, not automatic random cuts')
    if (len(entries) > 3 and len({s.get('camera','push_in') for s in entries}) == 1
            and all(s.get('camera','push_in') != 'static' and s.get('camera_strength',.35) > 0 for s in entries)):
        notes.append('Every photograph uses the same continuous camera move; consider authored rests or reframing')
    return {'seconds':round(seconds,6),'scene_count':len(entries),'long_photographic_scenes':long_holds,
            'notes':notes,'first_two_seconds_review_required':True,
            'character_and_prop_continuity_review_required':True,
            'ending_review_required':True,'automated_story_approval':False}


def render_scene(scene: dict, output: Path, w: int, h: int, fps: int,
                 subtitles_path: Path | None = None) -> None:
    if scene.get('motion_clips'):
        from .cinematic_edit import render
        receipt=render(scene,output,w,h,fps,subtitles_path)
        output.with_suffix('.edit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        return
    # Smooth photographic push-in, not an artificial skeleton/cutout animation.
    frames = frame_count(scene["seconds"], fps)
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if scene["motion_video"]:
        args += ["-i", str(scene["motion_video"])]
    else:
        from .smooth_camera import render_still
        camera_clip = output.with_suffix('.camera.mp4')
        render_still(scene['image'], camera_clip, w, h, fps, frames, scene.get('camera','push_in'),
                     strength=scene.get('camera_strength',.35), framing=scene.get('framing'))
        args += ['-i', str(camera_clip)]
    args += ["-i", str(scene["voice"])]
    with_music = scene["music"] is not None
    if with_music:
        args += ["-stream_loop", "-1", "-i", str(scene["music"])]
    if scene["motion_video"]:
        motion_filter = (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
                         f"crop={w}:{h},setsar=1,fps={fps},format=yuv420p[v]")
    else:
        motion_filter = '[0:v]setsar=1,format=yuv420p[v]'
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
                 "normalize=0,alimiter=limit=0.92:latency=1[a]")
    else:
        filt += ";[voice]alimiter=limit=0.92:latency=1[a]"
    args += ["-filter_complex", filt, "-map", "[v]", "-map", "[a]",
             "-frames:v", str(frames), "-c:v", "libx264", "-threads", "2", "-preset", "medium",
             "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(fps),
             "-c:a", "pcm_s16le" if output.suffix == ".mov" else "aac",
             "-b:a", "192k", "-ar", "48000", "-ac", "2",
             "-t", f"{frames/fps:.6f}", "-movflags", "+faststart", str(output)]
    try:
        _run(args)
    finally:
        if not scene['motion_video']:
            camera_clip.unlink(missing_ok=True)


def qa(video: Path, w: int, h: int, fps: int, *, entries: list[dict] | None = None) -> dict:
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
    decoded=subprocess.run(['ffmpeg','-hide_banner','-nostdin','-nostats','-v','info','-xerror',
        '-i',str(video),'-vf','blackdetect=d=0:pix_th=0.025:pic_th=0.995,freezedetect=n=-60dB:d=1.2','-f','null','-'],
        capture_output=True,text=True,timeout=180)
    if decoded.returncode:raise RuntimeError('Full final video decode failed: '+decoded.stderr[-1700:])
    black_intervals=[line.strip() for line in decoded.stderr.splitlines() if 'black_start:' in line]
    if black_intervals:raise RuntimeError('Final video contains black frames: '+'; '.join(black_intervals))
    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp) / "first.png"
        _run(["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", "1", str(frame)])
        with Image.open(frame) as im:
            mean = sum(ImageStat.Stat(im.convert("RGB")).mean) / 3
        if mean < 9:
            raise RuntimeError(f"First frame is near-black: brightness={mean:.1f}")
    from .delivery_qc import inspect_av, freeze_intervals
    av = inspect_av(video, info, fps)
    holds, offset = [], 0.
    for scene in entries or []:
        end = offset + frame_count(scene['seconds'], fps) / fps
        if not (scene.get('motion_video') or scene.get('motion_clips')) and (
                scene.get('camera') == 'static' or scene.get('camera_strength') == 0):
            holds.append((offset, end))
        offset = end
    freezes = freeze_intervals(decoded.stderr, av['video_clock_seconds'], holds)
    return {"technical_pass": True, "full_decode_pass": True,
            'full_timeline_black_frame_check':True,'black_intervals':[],
            "first_frame_mean": round(mean, 2),
            "width": w, "height": h, "fps": fps,
            "duration_s": float(info.get("format", {}).get("duration", 0)),
            "av_delivery": av, "measured_holds": freezes,
            "unexplained_holds_require_review": any(f['requires_visual_review'] for f in freezes),
            "human_artistic_review": "pending", "rights_review": "pending",
            "approved_for_publication": False}


def run(manifest: Path, destination: Path, *, smoke: bool = False,
        dry_run: bool = False, cache_only: bool = False,
        cache_scenes: set[int] | None = None) -> dict:
    item = validate_project(manifest, smoke=smoke)
    if cache_scenes is not None and (not cache_only or not cache_scenes or
            not cache_scenes.issubset({s['id'] for s in item['entries']})):
        raise Blocked('An explicit existing scene set is only valid for cache-only rendering')
    target = Path(destination).resolve() / item["id"]
    if target.exists():
        raise Blocked(f"Output exists; refuse silent overwrite: {target}")
    assets = asset_hashes(item)
    if item['master_audio']:
        length=duration(item['master_audio'])
        expected=sum(frame_count(s['seconds'],item['fps'])/item['fps'] for s in item['entries'])
        if abs(length-expected)>.08:
            raise Blocked('Mixed story audio differs from the full original narration timeline')
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
        cache=destination/'.scene_cache'/item['id'];cache.mkdir(parents=True,exist_ok=True)
        write_srt(item["entries"], staging / "subtitles.srt", item["fps"])
        write_ass(item['entries'], staging/'subtitles.ass', item['fps'],item['ai_disclosure'],item['intro_hook'])
        for index, scene in enumerate(item["entries"], 1):
            if cache_only and cache_scenes is not None and scene['id'] not in cache_scenes:
                continue
            scene_captions=clips/f'{index:03d}.ass'
            write_ass([scene],scene_captions,item['fps'],item['ai_disclosure'] if index==len(item['entries']) else None,item['intro_hook'] if index==1 else None)
            sources=[scene['image'],scene['voice'],scene['music'],scene['motion_video'],*scene['motion_clips']]
            fingerprint_data={'renderer':VERSION,'width':item['width'],'height':item['height'],
                'fps':item['fps'],'source_hashes':[sha(p) for p in sources if p],
                'captions':scene_captions.read_text(),'seconds':scene['seconds'],
                'camera':scene.get('camera','push_in'),'camera_strength':scene.get('camera_strength',.35),
                'framing':scene.get('framing')}
            if scene.get('foreground_soft_focus'):fingerprint_data['foreground_soft_focus']=scene['foreground_soft_focus']
            fingerprint=hashlib.sha256(json.dumps(fingerprint_data,sort_keys=True).encode()).hexdigest()
            # QuickTime keeps the exact video time base and lossless PCM.
            # AAC is encoded only once, when the complete film is delivered.
            cached=cache/f'{index:03d}_{fingerprint[:24]}.mov';proof=cached.with_suffix('.cache.json')
            valid=False
            if cached.is_file() and proof.is_file():
                saved=json.loads(proof.read_text())
                valid=(saved.get('fingerprint')==fingerprint and saved.get('sha256')==sha(cached)
                       and abs(video_duration(cached)-frame_count(scene['seconds'],item['fps'])/item['fps'])<.006)
            if not valid:
                building=cached.with_suffix('.building.mov')
                render_scene(scene,building,item['width'],item['height'],item['fps'],subtitles_path=scene_captions)
                os.replace(building,cached)
                if building.with_suffix('.edit.json').is_file():os.replace(building.with_suffix('.edit.json'),cached.with_suffix('.edit.json'))
                proof.write_text(json.dumps({'fingerprint':fingerprint,'sha256':sha(cached),'complete':True}))
            os.link(cached,clips/f'{index:03d}.mov')
            if cached.with_suffix('.edit.json').is_file():shutil.copy2(cached.with_suffix('.edit.json'),clips/f'{index:03d}.edit.json')
        if cache_only:
            return {'status':'SCENES_CACHED','story':item['id'],
                    'scenes':sorted(cache_scenes) if cache_scenes is not None else [s['id'] for s in item['entries']],
                    'final_movie_created':False,'approved_for_publication':False}
        playlist = staging / "clips.txt"
        playlist.write_text(
            "".join(f"file '{p.resolve()}'\n" for p in sorted(clips.glob("*.mov"))),
            encoding="utf-8")
        _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
              "-i", str(playlist), "-c", "copy", str(staging / "joined.mov")])
        # Captions were burned once per scene; the full portrait movie does
        # not need a second costly H.264 encode with the same subtitle pixels.
        from .remaster_audio import normalize
        mastering = normalize(item['master_audio'] or staging/'joined.mov', staging/'audio_master.wav',
                              lufs=-16, tp=-1.7, channels=2)
        args=['ffmpeg','-v','error','-nostdin','-y','-i',str(staging/'joined.mov'),
              '-i',str(staging/'audio_master.wav'),'-map','0:v:0','-map','1:a:0',
              '-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-ac','2']
        args+=['-movflags','+faststart',str(staging/'final.mp4')]
        _run(args)
        # Use the actual rendered composition. A separately center-cropped
        # JPG could otherwise lose the very prop the focal camera preserved.
        cover_time = 0.
        for scene in item['entries']:
            if scene['id'] == item['cover_scene_id']:break
            cover_time += frame_count(scene['seconds'],item['fps'])/item['fps']
        _run(['ffmpeg','-v','error','-nostdin','-y','-ss',f'{cover_time:.6f}',
              '-i',str(staging/'final.mp4'),'-frames:v','1',str(staging/'cover_source.png')])
        cover(staging/'cover_source.png',item['cover_text'],staging/'cover.jpg',item['width'],item['height'])
        report = qa(staging / "final.mp4", item["width"], item["height"], item["fps"], entries=item['entries'])
        from .delivery_qc import inspect_cover
        report['cover'] = inspect_cover(staging/'cover.jpg') if not smoke else {'smoke_only':True}
        report['audio_mastering'] = {'target_lufs':-16,'target_true_peak_dbtp':-1.7,
                                    'speech_speed_factor':1.0,'input_measurements':mastering,
                                    'scene_audio_codec':'pcm_s16le',
                                    'final_aac_generations':1,
                                    'limiter_latency_compensated':True}
        report['captions'] = [{'scene':s['id'],**s['caption_audit'],
                              'source_verified':s['caption_source_verified']} for s in item['entries']]
        report['composition'] = [{'scene':s['id'],'framing':s['framing'],
                                  'source_upscale':s['source_upscale']} for s in item['entries']]
        report['editorial'] = editorial_report(item['entries'],item['fps'])
        report['cover_source'] = {'encoded_final_frame':True,'scene':item['cover_scene_id'],
                                  'timestamp_s':round(cover_time,6)}
        expected=sum(frame_count(s['seconds'],item['fps'])/item['fps'] for s in item['entries'])
        if abs(report['duration_s']-expected)>.10:raise RuntimeError('Final movie changed the full original narration duration')
        report['full_original_narration_clock_s']=expected
        report['duration_matches_full_original_narration']=True
        report["smoke_only"] = smoke
        report["real_motion_video_scenes"] = sum(bool(s["motion_video"]) for s in item["entries"])
        report['neural_clip_montage_scenes']=sum(bool(s['motion_clips']) for s in item['entries'])
        report['editorial_shots']=[json.loads(p.read_text()) for p in sorted(clips.glob('*.edit.json'))]
        report["music_for_all_scenes"] = all(s["music"] for s in item["entries"])
        (staging / "qa_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        from .shorts_review import create_review, join_narration
        shutil.copyfile(item['script'],staging/'script_ru.txt')
        join_narration(item['entries'],staging/'narration_ru.wav',item['fps'])
        review = create_review(item,staging)
        if (sha(item['script']) != item['digest'] or sha(Path(manifest)) != item['project_sha256']
                or asset_hashes(item) != assets):
            raise Blocked('Source material or project changed during rendering; final export was not committed')
        receipt = {"schema": 1, "engine": VERSION, "story_id": item["id"],
                   'project_sha256':item['project_sha256'],
                   "script_sha256": item["digest"], "assets_sha256": assets,
                   "final_sha256": sha(staging / "final.mp4"),
                   "review_status": "REVIEW_REQUIRED",
                   "approved_for_publication": False,
                   'review':review,
                   'delivery_sha256':{p.relative_to(staging).as_posix():sha(p)
                       for p in sorted(staging.rglob('*')) if p.is_file()
                       and p.parent != clips and p.name not in ('joined.mov','clips.txt','cover_source.png')}}
        (staging / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.rmtree(clips)
        (staging / "joined.mov").unlink()
        (staging / "cover_source.png").unlink()
        playlist.unlink()
        os.rename(staging, target)
    return {"status": "REVIEW_REQUIRED", "story": item["id"],
            "files": [str(target / fn) for fn in (
                "final.mp4", "cover.jpg", "subtitles.srt", "qa_report.json", "receipt.json",
                'script_ru.txt','narration_ru.wav','audio_master.wav','timeline.json','review.html')],
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
