#!/usr/bin/env python3
"""Offline-safe motion-comic MVP renderer. Optional user art, no AI/image API calls."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from studio.cinematic import CinematicPanel
    from studio.premium_assets import validate_premium_scene, review_episode
except ModuleNotFoundError:
    from cinematic import CinematicPanel
    from premium_assets import validate_premium_scene, review_episode

try:
    from studio.comic_art import panel as comic_panel
except ModuleNotFoundError:
    # Enables direct execution with `python studio/render.py` as well as `-m`.
    from comic_art import panel as comic_panel

HERE = Path(__file__).resolve().parent.parent
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]
PALETTES = {
    "earth": ((7, 17, 34), (35, 75, 108), (115, 181, 209)),
    "noctaris": ((20, 8, 28), (79, 28, 57), (204, 91, 105)),
}
EFFECTS = {"rain", "mirror", "portal", "flash", "embers", "none"}


def validate_episode(data: dict, asset_root: Path) -> list[dict]:
    if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
        raise ValueError("Episode must contain a scenes list")
    scenes = data["scenes"]
    if not 1 <= len(scenes) <= 100:
        raise ValueError("Scene count must be 1..100")
    total = 0.0
    ids = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            raise ValueError("Every scene must be an object")
        sid = scene.get("id")
        if not isinstance(sid, str) or sid in ids or not sid:
            raise ValueError("Scene IDs must be unique nonempty strings")
        ids.add(sid)
        duration = scene.get("duration")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not 0.5 <= duration <= 30:
            raise ValueError(f"{sid}: duration must be 0.5..30 s")
        total += duration
        if scene.get("palette", "earth") not in PALETTES:
            raise ValueError(f"{sid}: unknown palette")
        if scene.get("effect", "none") not in EFFECTS:
            raise ValueError(f"{sid}: unknown effect")
        if any(not isinstance(scene.get(k, ""), str) for k in ("heading", "caption")):
            raise ValueError(f"{sid}: headings and captions must be text")
        art = scene.get("art")
        if art is not None and art not in (
            "earth_photo", "earth_mirror", "two_worlds",
            "exchange", "fortress", "sunrise"
        ):
            raise ValueError(f"{sid}: unsupported comic scene art")
        if "cinematic" in scene:
            validate_premium_scene(scene,asset_root,publish=False,min_width=256,min_height=448)
        bg = scene.get("background")
        if bg is not None:
            if not isinstance(bg, str):
                raise ValueError(f"{sid}: background must be a path string")
            p = (asset_root / bg).resolve()
            if asset_root.resolve() not in p.parents:
                raise ValueError(f"{sid}: artwork path is outside assets/")
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                raise ValueError(f"{sid}: unsupported artwork type")
            if not p.is_file():
                raise FileNotFoundError(p)
    if total > 600:
        raise ValueError("Episode exceeds 10-minute renderer limit")
    return scenes


def font(size: int):
    for p in FONT_PATHS:
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def gradient(size: tuple[int, int], palette: str) -> Image.Image:
    w, h = size
    top, mid, bottom = PALETTES[palette]
    y = np.linspace(0.0, 1.0, h)[:, None]
    stops = []
    for a, b, c in zip(top, mid, bottom):
        v = np.where(y < 0.55,
                     a + (b-a)*(y / 0.55),
                     b + (c-b)*((y - 0.55)/0.45))
        stops.append(np.broadcast_to(v, (h, w)))
    return Image.fromarray(np.uint8(np.clip(np.stack(stops, axis=2), 0, 255)), "RGB")


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, face, max_width: int) -> list[str]:
    words = text.strip().split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=face)[2] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def centered_text(draw, text, face, center_y, width, fill, max_width,
                  max_height=None, center_x=None):
    """Mobile-safe text fitting, avoiding the right-side social-app controls.

    Never silently clip long captions. Shrink font and wrap at up to four lines;
    reject unrenderable text rather than drawing outside safe bounds.
    """
    if not text:
        return
    max_height = max_height or 100000
    center_x = center_x if center_x is not None else width / 2
    size = getattr(face, "size", 18)
    best = None
    for font_size in range(max(10, size), 9, -1):
        font_now = font(font_size)
        lines = []
        for para in text.strip().splitlines():
            line = ""
            for word in para.split():
                proposal = (line + " " + word).strip()
                if line and draw.textbbox((0, 0), proposal, font=font_now)[2] > max_width:
                    lines.append(line)
                    line = word
                else:
                    line = proposal
                # Word longer than the available width? Split safely into chunks.
                if line and draw.textbbox((0, 0), line, font=font_now)[2] > max_width:
                    chunk = ""
                    for char in line:
                        next_chunk = chunk + char
                        if chunk and draw.textbbox((0, 0), next_chunk, font=font_now)[2] > max_width:
                            lines.append(chunk)
                            chunk = char
                        else:
                            chunk = next_chunk
                    line = chunk
            if line:
                lines.append(line)
        bbox = draw.textbbox((0, 0), "Ag", font=font_now)
        line_h = bbox[3] - bbox[1] + max(3, int(font_size * 0.28))
        if len(lines) <= 4 and line_h * len(lines) <= max_height:
            best = (font_now,lines,line_h,font_size)
            break
    if best is None:
        raise ValueError("Caption cannot fit into safe text area")
    face_now,lines,line_h,font_size = best
    y0 = center_y - line_h * len(lines) / 2
    stroke = max(1,round(font_size*.11))
    for index,line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=face_now)
        tw = bbox[2]-bbox[0]
        x = round(center_x - tw/2)
        y = round(y0 + index*line_h)
        draw.text((x,y),line,font=face_now,fill=fill,
                  stroke_width=stroke,stroke_fill=(0,0,0))


def animate_frame(base, scene: dict, t: float, frame_index: int, fps: int) -> Image.Image:
    w, h = base.size
    # Subtle breathing camera movement; crop from a slightly enlarged source.
    z = 1.04 + 0.025 * t
    zw, zh = math.ceil(w*z), math.ceil(h*z)
    enlarged = base.resize((zw, zh), Image.Resampling.BILINEAR)
    dx = round(math.sin(t*1.4) * w * .006)
    dy = round(math.cos(t*.9) * h * .003)
    x = max(0, min(zw-w, (zw-w)//2 + dx))
    y = max(0, min(zh-h, (zh-h)//2 + dy))
    img = enlarged.crop((x, y, x+w, y+h)).convert("RGBA")
    fx = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(fx)
    effect = scene.get("effect", "none")
    unit = w / 540
    if effect in ("mirror", "portal", "flash"):
        cx, cy = round(w*.51), round(h*.41)
        rx, ry = round(w*.27), round(h*.30)
        for n in range(9, 0, -1):
            pad = round(n*unit*3)
            opacity = int((12-n) * (1 + math.sin(t*3.5)**2) * 3)
            d.ellipse((cx-rx-pad, cy-ry-pad, cx+rx+pad, cy+ry+pad),
                      outline=(112, 200, 246, min(90, opacity)), width=max(1, round(2*unit)))
        if effect == "portal":
            for k in range(24):
                a = k*2.399 + frame_index*.05
                r = (.2 + (k % 7)/12)*rx
                px = round(cx + math.cos(a)*r)
                py = round(cy + math.sin(a)*r*1.2)
                sz = max(1, round((2 + k%3)*unit))
                d.ellipse((px-sz,py-sz,px+sz,py+sz),fill=(220,233,255,130))
    if effect == "rain":
        for k in range(95):
            seed = (k*2654435761) & 0xffffffff
            px = (seed % w + frame_index*(4+k%4)) % w
            py = ((seed // 997)%h + frame_index*(11+k%6)) % h
            d.line((px, py, px-max(2,int(6*unit)), py+max(6,int(22*unit))),
                   fill=(175, 215, 255, 55), width=max(1, int(unit)))
    if effect == "embers":
        for k in range(70):
            px = (k*103 + int(math.sin(t+k)*11))%w
            py = (k*177 - int(t*(16+k%7)))%h
            r = max(1,round((k%3+1)*unit))
            d.ellipse((px-r,py-r,px+r,py+r), fill=(255, 154, 92, 90))
    img = Image.alpha_composite(img, fx)
    duration = float(scene["duration"])
    if effect == "flash":
        brightness = int(180 * max(0, 1 - abs(t-0.55) / 0.19))
        img = Image.alpha_composite(img, Image.new("RGBA", (w, h), (225,235,255,brightness)))
    # Bottom-safe black caption band; no text in lowest 15% where TikTok controls overlap.
    band = Image.new("RGBA", (w, h), (0,0,0,0))
    bd = ImageDraw.Draw(band)
    bd.rectangle((0,round(h*.59),w,round(h*.85)),fill=(2,6,15,155))
    img = Image.alpha_composite(img, band).convert("RGB")
    draw = ImageDraw.Draw(img)
    heading = scene.get("heading", "")
    caption = scene.get("caption", "")
    centered_text(draw, heading, font(max(13, round(w*.055))), h*.665, w,
                  (255,236,209), round(w*.70), max_height=round(h*.105),
                  center_x=w*.45)
    centered_text(draw, caption, font(max(11, round(w*.033))), h*.772, w,
                  (219,232,247), round(w*.70), max_height=round(h*.135),
                  center_x=w*.45)
    # Keep fades short to avoid sluggish comic pacing.
    fade = min(1.0, t/.18, (duration-t)/.18)
    fade = max(.0, fade)
    if fade < 1:
        img = Image.blend(Image.new("RGB", (w,h),(2,5,10)), img, fade)
    return img


def make_audio(path: Path, total_duration: float):
    """Generated low-volume dark-ambient pad; not dialogue or a music license."""
    sr = 22050
    n = round(total_duration*sr)
    time = np.arange(n, dtype=np.float64)/sr
    layer = (np.sin(2*np.pi*73.42*time)+0.4*np.sin(2*np.pi*110.0*time)
             +0.22*np.sin(2*np.pi*146.83*time))
    breath = .55 + .45*np.sin(2*np.pi*.17*time)**2
    intro = np.minimum(time/.4, 1)
    outro = np.minimum((total_duration-time)/.55,1)
    audio = layer * breath * np.maximum(0, np.minimum(intro,outro)) * .065
    pcm = np.int16(np.clip(audio,-1,1)*32767)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def render(episode: Path, output: Path, assets: Path, width: int, height: int, fps: int,
           quality: str = "preview"):
    if quality not in ("preview","master"):
        raise ValueError("Unknown encoder quality preset")
    if quality == "master":
        if width < 1080 or height < 1920 or fps < 24:
            raise ValueError("Master output requires at least 1080x1920 at 24 fps")
        src = json.loads(episode.read_text(encoding="utf-8"))
        review_episode(src,assets,publish=True)
        from studio.work_contract import verify_work_lock
        from studio.identity_qc import verify_episode_characters
        from studio.audio_master import audit_audio
        verify_work_lock(src,strict=True)
        verify_episode_characters(src,assets,strict=True)
        audit_audio(src,assets,publish=True,fps=fps)
        from studio.animation_quality import audit_performance
        audit_performance(src,publish=True)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required and must be on PATH")
    if width < 64 or height < 64 or fps < 1 or fps > 60:
        raise ValueError("Invalid dimensions or frame rate")
    with episode.open("r", encoding="utf-8") as f:
        data = json.load(f)
    scenes = validate_episode(data, assets)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        silent = work / "video.mp4"
        audio = work / "ambient.wav"
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "slow" if quality == "master" else "veryfast",
               "-crf", "18" if quality == "master" else "23",
               "-pix_fmt", "yuv420p", str(silent)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        total_frames = 0
        try:
            for scene in scenes:
                premium = CinematicPanel(scene,assets,width,height) if scene.get("cinematic") else None
                palette = scene.get("palette","earth")
                if scene.get("background"):
                    with Image.open(assets / scene["background"]) as original:
                        base = cover(original, (width, height))
                elif scene.get("art"):
                    base = comic_panel(scene, width, height)
                else:
                    base = gradient((width,height),palette)
                frames = max(1,round(float(scene["duration"])*fps))
                for frame in range(frames):
                    # Cinematic scene layers replace procedural stand-ins entirely.
                    frame_base = premium.frame(frame/max(1,frames-1)) if premium else base
                    image = animate_frame(frame_base, scene, frame/fps, frame, fps)
                    proc.stdin.write(image.tobytes())
                total_frames += frames
            proc.stdin.close()
            error = proc.stderr.read().decode("utf-8", errors="replace")
            code = proc.wait()
            if code:
                raise RuntimeError(f"ffmpeg video failure ({code}): {error}")
        except Exception:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            raise
        actual_duration = total_frames / fps
        make_audio(audio, actual_duration)
        mux = ["ffmpeg","-nostdin","-v","error","-y","-i",str(silent),
               "-i",str(audio),"-map","0:v:0","-map","1:a:0","-c:v","copy",
               "-c:a","aac","-b:a","192k" if quality == "master" else "96k",
               "-ar","48000" if quality == "master" else "22050",
               "-ac","2" if quality == "master" else "1",
               "-shortest","-movflags","+faststart",str(output)]
        subprocess.run(mux, check=True)
    return {"output":str(output),"frames":total_frames,"duration_seconds":actual_duration}


def main():
    p=argparse.ArgumentParser(description="Render an original motion comic without paid tools")
    p.add_argument("--episode",type=Path,default=HERE/"episodes/E01_THE_REFLECTION.json")
    p.add_argument("--output",type=Path,default=HERE/"dist/E01_preview.mp4")
    p.add_argument("--assets",type=Path,default=HERE/"assets")
    p.add_argument("--width",type=int,default=540)
    p.add_argument("--height",type=int,default=960)
    p.add_argument("--fps",type=int,default=12)
    p.add_argument("--quality",choices=("preview","master"),default="preview")
    args=p.parse_args()
    result=render(args.episode,args.output,args.assets,args.width,args.height,args.fps,
                  quality=args.quality)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()
