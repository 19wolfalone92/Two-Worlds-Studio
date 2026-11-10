"""Offline delivery measurements and a review board from the actual MP4.

Measures decoded frame timestamps and the encoded AAC, not source intentions.
Technical checks never certify acting, anatomy, voice pronunciation or retention.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageStat


def _run(args: list[str], *, timeout: int = 180, text: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=text, timeout=timeout)
    if result.returncode:
        error = result.stderr if text else result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError("Delivery measurement failed: " + error[-1500:])
    return result


def loudness(video: Path) -> dict:
    result = _run(["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-v", "info",
                   "-i", str(video), "-map", "0:a:0", "-vn", "-af",
                   "loudnorm=I=-16:TP=-1.7:LRA=9:print_format=json", "-f", "null", "-"])
    matches = re.findall(r'\{\s*"input_i".*?\}', result.stderr, re.S)
    if not matches:
        raise RuntimeError("Encoded audio loudness measurement is missing")
    measured = json.loads(matches[-1])
    values = {"integrated_lufs": float(measured["input_i"]),
              "true_peak_dbtp": float(measured["input_tp"]),
              "loudness_range_lu": float(measured["input_lra"])}
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError("Encoded audio is silent or has no finite loudness")
    return values


def dialogue_clock(video: Path, shots: list[dict], fps: int) -> list[dict]:
    """Locate audible, locked WAVs in the encoded mix using a bounded waveform search."""
    locked = [(i + 1, shot) for i, shot in enumerate(shots)
              if shot.get("dialogue_text") and shot.get("voice")]
    if not locked:
        return []
    rate = 8000

    def pcm(path):
        result = _run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
                       "-af", f"aresample={rate},highpass=f=120", "-ac", "1",
                       "-f", "s16le", "-"], text=False)
        return np.frombuffer(result.stdout, dtype="<i2")

    mix = pcm(video)
    tolerance = round(rate / fps)
    reports = []
    for index, shot in locked:
        reference = pcm(shot["voice"]).astype(np.float64)
        expected = shot["start"] + shot["voice_offset"]
        result = {"shot": index, "expected_start_seconds": round(expected, 4),
                  "status": "REVIEW REQUIRED", "reason": "waveform_match_unverified"}
        reports.append(result)
        length = min(len(reference), round(rate * .6))
        if length < rate * .12:
            continue
        energy = np.cumsum(np.r_[0., reference ** 2])
        anchor = int(np.argmax(energy[length:] - energy[:-length]))
        template = reference[anchor:anchor + length].copy()
        template -= template.mean()
        strength = np.linalg.norm(template)
        center = round(expected * rate) + anchor
        left = max(0, center - 5 * rate)
        right = min(len(mix) - length, center + 5 * rate)
        if strength < 10 * math.sqrt(length) or right < left:
            continue
        excerpt = mix[left:right + length].astype(np.float64)
        fft_size = 1 << (len(excerpt) + length - 2).bit_length()
        cross = np.fft.irfft(np.fft.rfft(excerpt, fft_size) *
                             np.fft.rfft(template, fft_size).conj())[:right - left + 1]
        energy = np.cumsum(np.r_[0., excerpt ** 2])
        scores = cross / np.maximum(1., strength * np.sqrt(energy[length:] - energy[:-length]))
        offsets = np.arange(left, right + 1) - center
        nearby = np.abs(offsets) <= tolerance
        if np.any(nearby) and scores[nearby].max() >= .35:
            best = int(np.argmax(np.where(nearby, scores, -np.inf)))
            result.update(status="PASS", reason="waveform_within_one_frame")
        else:
            best = int(scores.argmax())
            if scores[best] >= .70:
                result.update(status="BLOCKED", reason="waveform_outside_one_frame")
        result["correlation"] = round(float(scores[best]), 4)
        if result["status"] != "REVIEW REQUIRED":
            result["offset_ms"] = round(1000 * int(offsets[best]) / rate, 3)
    return reports


def inspect_av(video: Path, info: dict, fps: int, *, strict_loudness: bool = True) -> dict:
    """Catch VFR hidden by an average of 30, silent AAC and A/V edge drift."""
    if type(fps) is not int or not 1 <= fps <= 120:
        raise ValueError('Delivery FPS must be an integer within 1..120')
    streams = info.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise RuntimeError("Delivery needs exactly one video and one audio stream")
    v, a = videos[0], audios[0]
    if v.get('sample_aspect_ratio') not in (None,'1:1','0:1','N/A'):
        raise RuntimeError('Non-square pixels change the displayed portrait composition')
    for data in v.get('side_data_list', []):
        if float(data.get('rotation',0)) % 360:
            raise RuntimeError('Video rotation metadata changes the displayed portrait orientation')
    if int(a.get("sample_rate", 0)) != 48000 or a.get("channels") != 2:
        raise RuntimeError("Delivery audio must be 48 kHz stereo AAC")
    result = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                   "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)])
    frames = json.loads(result.stdout).get("frames", [])
    try:
        times = [float(f["best_effort_timestamp_time"]) for f in frames]
    except (ValueError, KeyError) as exc:
        raise RuntimeError("Decoded frame timestamps are missing") from exc
    if not times or not all(math.isfinite(t) for t in times):
        raise RuntimeError("No finite decoded video clock")
    worst = max((abs(t - (times[0] + i / fps)) for i, t in enumerate(times)), default=0)
    if any(b <= a for a, b in zip(times, times[1:])) or worst > .001:
        raise RuntimeError("Decoded video is not constant-frame-rate or has missing/repeated timestamps")
    if v.get("nb_frames") is not None and int(v["nb_frames"]) != len(times):
        raise RuntimeError("Declared frame count differs from decoded frames")
    video_start, video_end = times[0], times[0] + len(times) / fps
    audio_start = float(a.get("start_time", 0))
    try:
        audio_end = audio_start + float(a["duration"])
    except (ValueError, KeyError) as exc:
        raise RuntimeError("AAC stream duration is missing") from exc
    if not math.isfinite(audio_start) or not math.isfinite(audio_end):
        raise RuntimeError("AAC stream clock is not finite")
    start_difference = abs(audio_start - video_start)
    end_difference = abs(audio_end - video_end)
    if abs(video_start) > .05 or start_difference > .08 or end_difference > .12:
        raise RuntimeError("Audio/video start or end drift exceeds the delivery tolerance")
    levels = loudness(video)
    if levels["true_peak_dbtp"] > -1.0:
        raise RuntimeError("Encoded AAC true peak exceeds -1 dBTP")
    target_ok = -18 <= levels["integrated_lufs"] <= -14
    if strict_loudness and not target_ok:
        raise RuntimeError("Encoded AAC is outside the -18..-14 LUFS delivery range")
    return {"decoded_frames": len(times), "constant_frame_rate_verified": True,
            "maximum_frame_clock_error_s": round(worst, 7),
            "video_clock_seconds": round(video_end - video_start, 6),
            "audio_start_difference_s": round(start_difference, 6),
            "audio_end_difference_s": round(end_difference, 6),
            "encoded_audio": {**levels, "target_loudness_pass": target_ok},
            "speech_sync_listening_required": True}


def freeze_intervals(log: str, duration: float, holds: list[tuple[float, float]]) -> list[dict]:
    """Classify measured holds without mistaking a chosen still for a broken clip."""
    intervals, start = [], None
    for match in re.finditer(r"freeze_(start|end):\s*([0-9.]+)", log):
        kind, value = match.group(1), float(match.group(2))
        if kind == "start":
            start = value
        elif start is not None:
            intervals.append((start, value))
            start = None
    if start is not None:
        intervals.append((start, duration))
    result = []
    for begin, end in intervals:
        # Union coverage avoids counting overlapping authored hold ranges twice.
        cursor, coverage = begin, 0.
        for left, right in sorted(holds):
            left, right = max(begin, left, cursor), min(end, right)
            if right > left:
                coverage += right - left
                cursor = right
        intended = coverage >= max(0., end - begin - .10)
        result.append({"start": round(begin, 4), "end": round(end, 4),
                       "seconds": round(end - begin, 4), "authored_hold": intended,
                       "requires_visual_review": not intended})
    return result


def inspect_cover(path: Path) -> dict:
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source).convert("RGB")
        width, height = im.size
        thumb = im.resize((108, 192))
        mean = sum(ImageStat.Stat(thumb).mean) / 3
        saturation = ImageStat.Stat(thumb.convert("HSV")).mean[1]
    if mean < 9:
        raise RuntimeError("Cover image is near-black")
    if width < 540 or height < 960 or abs(width / height - 9 / 16) > .005:
        raise RuntimeError("Cover must be a real portrait image of at least 540x960")
    return {"width": width, "height": height, "mean_brightness": round(mean, 2),
            "mean_saturation": round(saturation, 2), "local_pixels_verified": True,
            "destination_cover_verified": False,
            "destination_preview_check_required": True}


def review_board(video: Path, output: Path, seconds: float, *, samples: int = 9) -> dict:
    from .caption_layout import font
    if type(samples) is not int or not 3 <= samples <= 18 or seconds <= 0:
        raise ValueError("Review board needs a real duration and 3..18 samples")
    w, h, gap, footer, columns = 270, 480, 12, 42, 3
    rows = math.ceil(samples / columns)
    board = Image.new("RGB", (columns * (w + gap) + gap, rows * (h + footer + gap) + gap), "#121b27")
    draw = ImageDraw.Draw(board)
    times = [min(max(0., seconds - .08), seconds * i / (samples - 1)) for i in range(samples)]
    with tempfile.TemporaryDirectory(prefix="tw-review-") as temp:
        for index, timestamp in enumerate(times):
            frame = Path(temp) / f"frame_{index}.png"
            _run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{timestamp:.6f}",
                  "-i", str(video), "-frames:v", "1", "-vf", f"scale={w}:{h}", str(frame)])
            with Image.open(frame) as image:
                tile = image.convert("RGB")
            x, y = gap + index % columns * (w + gap), gap + index // columns * (h + footer + gap)
            board.paste(tile, (x, y))
            draw.text((x + 4, y + h + 5), f"{timestamp:05.2f} s  |  MP4", font=font(17), fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output)
    return {"samples_seconds": [round(t, 3) for t in times],
            "source": "decoded_final_mp4", "artistic_approval": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--cover", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-board", type=Path)
    args = parser.parse_args(argv)
    from .shorts_factory import qa
    report = qa(args.video, args.width, args.height, args.fps)
    if args.cover:
        report["cover"] = inspect_cover(args.cover)
    if args.review_board:
        report["review_board"] = review_board(args.video, args.review_board, report["duration_s"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"technical_pass": report["technical_pass"],
                      "duration_s": report["duration_s"],
                      "approved_for_publication": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
