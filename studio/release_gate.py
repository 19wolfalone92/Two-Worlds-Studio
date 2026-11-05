"""Read-only release verification for STORY GLITCH vertical MP4 masters.

No network, AI generation, publishing, or fabricated approval.
Distinguishes engineering correctness from subjective film quality.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import re
import subprocess
from pathlib import Path
from fractions import Fraction

from PIL import Image, ImageOps, ImageStat

TARGET = (1080, 1920, 30)
TIME = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")
CUE = re.compile(r"(?m)^(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=index,codec_name,codec_type,width,height,avg_frame_rate,r_frame_rate",
        "-of", "json", str(path)
    ]
    value = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    return json.loads(value.stdout)


def _duration(streams: dict) -> float:
    value = float(streams.get("format", {}).get("duration", 0))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Missing positive MP4 duration")
    return value


def _ms(text: str) -> int:
    match = TIME.fullmatch(text)
    if not match:
        raise ValueError("Invalid SRT timestamp")
    hh, mm, ss, milli = map(int, match.groups())
    if mm > 59 or ss > 59:
        raise ValueError("Invalid SRT clock")
    return ((hh * 60 + mm) * 60 + ss) * 1000 + milli


def inspect_subtitles(path: Path, duration: float, *, require_russian: bool = True) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    cues = list(CUE.finditer(text))
    if not cues:
        raise ValueError("Missing valid numbered SRT cues")
    if require_russian and not re.search(r"[А-Яа-яЁё]", text):
        raise ValueError("Russian subtitle text is absent")
    previous_end = 0
    for i, match in enumerate(cues, 1):
        n, a, b = match.groups()
        start, end = _ms(a), _ms(b)
        if int(n) != i or start < previous_end or end <= start:
            raise ValueError("Disordered/overlapping subtitle cues")
        if end > duration * 1000 + 200:
            raise ValueError("Subtitles extend beyond the video duration")
        segment = text[match.end() : (cues[i].start() if i < len(cues) else len(text))].strip()
        if not segment:
            raise ValueError("Empty subtitle cue")
        previous_end = end
    return {"cues": len(cues), "last_cue_end_ms": previous_end}


def _grayscale_stats(img: Image.Image) -> dict:
    frame = ImageOps.exif_transpose(img).convert("L")
    frame.thumbnail((200, 360), Image.Resampling.BOX)
    hist = frame.histogram()
    size = sum(hist)
    if not size:
        raise ValueError("Empty image")
    dark = sum(hist[:12]) / size
    bright = sum(hist[246:]) / size
    mean = sum(i * c for i, c in enumerate(hist)) / size
    return {"mean_luma": round(mean, 2),
            "near_black_ratio": round(dark, 4),
            "near_white_ratio": round(bright, 4),
            "first_frame_valid": dark < .965 and bright < .965}


def cover_check(path: Path) -> dict:
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        size = img.size
        stats = _grayscale_stats(img)
    if size[0] < 540 or size[1] < 960:
        raise ValueError("Cover is below minimum 540x960")
    if abs(size[0] / size[1] - 9 / 16) > .02:
        raise ValueError("Cover aspect ratio is not 9:16")
    if not stats["first_frame_valid"]:
        raise ValueError("Nearly all-black or all-white cover rejected")
    return {"width": size[0], "height": size[1], **stats, "sha256": sha256(path)}


def first_frame_check(video: Path) -> dict:
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-frames:v", "1", "-vf", "scale=180:320",
        "-f", "image2pipe", "-vcodec", "png", "-"
    ]
    output = subprocess.run(command, capture_output=True, check=True, timeout=45).stdout
    if len(output) < 100:
        raise ValueError("Missing decodable first video frame")
    with Image.open(io.BytesIO(output)) as img:
        stats = _grayscale_stats(img)
    if not stats["first_frame_valid"]:
        raise ValueError("Video begins with an almost black/white frame")
    return stats


def check_video(video: Path, srt: Path, cover: Path,
                *, require_russian: bool = True, full_decode: bool = False) -> dict:
    for asset in (video, srt, cover):
        if not asset.is_file() or asset.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/nonempty deliverable: {asset.name}")
    data = ffprobe(video)
    duration = _duration(data)
    v = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    a = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if len(v) != 1 or len(a) != 1 or v[0].get("codec_name") != "h264" or a[0].get("codec_name") != "aac":
        raise ValueError("Expected one H.264 video stream and one AAC audio stream")
    if (v[0].get("width"), v[0].get("height")) != TARGET[:2]:
        raise ValueError("Video must be 1080x1920")
    fps = Fraction(str(v[0].get("avg_frame_rate", "0/1")))
    if fps <= 0 or abs(float(fps) - TARGET[2]) > .02:
        raise ValueError("Expected 30 FPS constant-rate master")
    subtitle_info = inspect_subtitles(srt, duration, require_russian=require_russian)
    cover_info = cover_check(cover)
    first = first_frame_check(video)
    if full_decode:
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error",
                        "-xerror", "-i", str(video), "-map", "0:v:0",
                        "-f", "null", "-"],
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=420)
    return {
        "status": "technical_pass", "sha256": sha256(video),
        "duration_seconds": round(duration, 3), "resolution": list(TARGET[:2]),
        "fps": round(float(fps), 3), "video_codec": "h264", "audio_codec": "aac",
        "subtitles": subtitle_info, "cover": cover_info, "first_frame": first,
        "full_decode_tested": bool(full_decode),
        "human_visual_audio_rights_approval_required": True,
        "published": False
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only technical QA; never publishes")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--srt", type=Path, required=True)
    parser.add_argument("--cover", type=Path, required=True)
    parser.add_argument("--language", choices=("ru", "en"), default="ru")
    parser.add_argument("--full-decode", action="store_true")
    args = parser.parse_args()
    result = check_video(args.video, args.srt, args.cover,
                         require_russian=args.language == "ru", full_decode=args.full_decode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
