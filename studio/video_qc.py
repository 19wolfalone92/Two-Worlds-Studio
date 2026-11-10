"""CPU-friendly automatic video/audio diagnostics, NOT professional aesthetic proof."""
from __future__ import annotations
import argparse
import json
import math
import re
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image


def _crop_pixels(box, width, height):
    x0, y0, x1, y1 = box
    x, y = round(x0 * width), round(y0 * height)
    w, h = round(x1 * width) - x, round(y1 * height) - y
    if min(w, h) < 24 or w * h > 650_000:
        raise ValueError("Temporal QC crop must be at least 24 px per side and at most 650,000 pixels")
    return x, y, w, h


def validate_regions(regions: list, size: tuple[int, int] | None = None) -> list[dict]:
    """Authored areas in decoded shot coordinates, before captions."""
    if not isinstance(regions, list) or not 1 <= len(regions) <= 6:
        raise ValueError("Specify one to six temporal QC regions")
    result, ids = [], set()
    for region in regions:
        if (not isinstance(region, dict) or
                set(region) - {"id", "kind", "box", "motion_required", "allow_exposure_change"}):
            raise ValueError("Invalid temporal QC region")
        name, kind, box = region.get("id"), region.get("kind"), region.get("box")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", name)
                or name in ids or kind not in ("face", "hands", "prop", "background")
                or not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in box)
                or box[2] - box[0] < .025 or box[3] - box[1] < .025
                or any(type(region.get(key, False)) is not bool
                       for key in ("motion_required", "allow_exposure_change"))):
            raise ValueError("Invalid temporal QC region geometry or policy")
        ids.add(name)
        if size is not None:
            _crop_pixels(box, *size)
        result.append({"motion_required": False, "allow_exposure_change": False, **region})
    return result


def _register(previous, current, window):
    dims = window.shape[::-1]
    small = [np.asarray(Image.fromarray(a).resize(dims, Image.Resampling.BOX), dtype=np.float32)
             for a in (previous, current)]
    if min(a.std() for a in small) < 2.5:
        return None
    left, right = [np.fft.rfft2((a - a.mean()) * window) for a in small]
    product = right * left.conj()
    magnitude = np.abs(product)
    phase = np.divide(product, magnitude, out=np.zeros_like(product), where=magnitude > 1e-6)
    correlation = np.fft.irfft2(phase, s=window.shape, axes=(0, 1))
    y, x = np.unravel_index(correlation.argmax(), correlation.shape)
    peak = float(correlation[y, x])
    if peak < .14:
        return None

    def offset(values, index):
        a, b = values[(index - 1) % len(values)], values[(index + 1) % len(values)]
        denom = a - 2 * peak + b
        fraction = .5 * (a - b) / denom if denom < -1e-9 else 0.
        base = index if index <= len(values) // 2 else index - len(values)
        return float(base + np.clip(fraction, -.5, .5))

    dx, dy = offset(correlation[y], x), offset(correlation[:, x], y)
    if abs(dx) > .12 * dims[0] or abs(dy) > .12 * dims[1]:
        return None
    return dx * previous.shape[1] / dims[0], dy * previous.shape[0] / dims[1]


def _pair(previous, current, window):
    raw = np.abs(current.astype(np.int16) - previous)
    held = raw.mean() <= .10 and np.count_nonzero(raw > 1) <= .001 * raw.size
    transform = _register(previous, current, window)
    if transform is None:
        return held, None, None
    dx, dy = transform
    height, width = previous.shape
    mx, my = math.ceil(abs(dx)) + 2, math.ceil(abs(dy)) + 2
    warped = Image.fromarray(current).transform(
        (width, height), Image.Transform.AFFINE, (1., 0., dx, 0., 1., dy),
        resample=Image.Resampling.BILINEAR,
    )
    # Exclude newly exposed borders and compare native pixels.
    interior = (slice(my, height - my, 2), slice(mx, width - mx, 2))
    delta = np.asarray(warped, dtype=np.int16)[interior] - previous[interior]
    luma = float(np.median(delta))
    return held, luma, float(np.percentile(np.abs(delta - luma), 90))


def _runs(values: list[bool], minimum: int):
    start = None
    for index, active in enumerate([*values, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= minimum:
                yield start, index - 1
            start = None


def _events(rows, region, fps):
    events = []

    def append(code, severity, start, last):
        events.append({"code": code, "severity": severity, "first_frame": start,
                       "last_frame": last + 1, "start_seconds": round(start / fps, 4),
                       "end_seconds": round((last + 2) / fps, 4)})

    if region["motion_required"]:
        for start, end in _runs([x[0] for x in rows], max(3, math.ceil(.24 * fps))):
            append("unexpected_hold", "BLOCKED", start, end)
    if not region["allow_exposure_change"]:
        luma = [x[1] or 0 for x in rows]
        reversals = [a * b < 0 and min(abs(a), abs(b)) >= 6
                     for a, b in zip(luma, luma[1:])]
        for start, end in _runs(reversals, 1):
            severe = end - start >= 3 and min(abs(x) for x in luma[start:end + 2]) >= 12
            append("exposure_oscillation", "BLOCKED" if severe else "REVIEW REQUIRED",
                   start, end + 1)
    for start, end in _runs([(x[2] or 0) >= 20 for x in rows], 3):
        append("unexplained_local_change", "REVIEW REQUIRED", start, end)
    if sum(x[1] is not None for x in rows) < .5 * len(rows):
        append("low_registration_confidence", "REVIEW REQUIRED", 0, len(rows) - 1)
    return events


def inspect_temporal(video: Path, regions: list, width: int, height: int,
                     fps: float, expected_frames: int) -> dict:
    """Check every frame of a bounded shot in the specified native crops."""
    if (type(expected_frames) is not int or not 2 <= expected_frames <= 1800 or
            not 1 <= fps <= 60 or not 64 <= width <= 4096 or not 64 <= height <= 4096):
        raise ValueError("Temporal QC requires a bounded shot and a known frame clock")
    regions = validate_regions(regions, (width, height))
    reports = []
    for region in regions:
        x, y, w, h = _crop_pixels(region["box"], width, height)
        factor = min(1., 160 / max(w, h))
        window = np.outer(np.hanning(max(12, round(h * factor))),
                          np.hanning(max(12, round(w * factor))))
        args = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "2",
                "-i", str(video), "-map", "0:v:0", "-an",
                "-vf", f"crop={w}:{h}:{x}:{y}:exact=1,format=gray",
                "-fps_mode", "passthrough", "-frames:v", str(expected_frames + 1),
                "-threads:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
        rows = []
        previous = None
        count = 0
        with tempfile.TemporaryFile() as errors, subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=errors,
        ) as proc:
            try:
                assert proc.stdout is not None
                while raw := proc.stdout.read(w * h):
                    if len(raw) != w * h:
                        raise ValueError("Truncated frame in temporal QC")
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
                    if previous is not None:
                        rows.append(_pair(previous, frame, window))
                    previous = frame
                    count += 1
                if proc.wait(timeout=15) or count != expected_frames:
                    raise ValueError(f"Temporal QC decoded {count} frames, expected {expected_frames}")
            except BaseException:
                proc.kill()
                raise
        reports.append({"id": region["id"], "kind": region["kind"], "crop_pixels": [w, h],
                        "frames_checked": count, "registered_pairs": sum(x[1] is not None for x in rows),
                        "events": _events(rows, region, fps)})
    severities = {event["severity"] for report in reports for event in report["events"]}
    status = ("BLOCKED" if "BLOCKED" in severities else
              "REVIEW REQUIRED" if severities else "PASS")
    return {"status": status, "method": "native_roi_translation_compensated_luma",
            "regions": reports, "anatomy_requires_review": True}


def sample_video(video:Path, *, fps:float=2.0, sample_width:int=72,
                 sample_height:int=128,max_frames:int=1200)->dict:
    if not 0<fps<=8:
        raise ValueError("Diagnostic sample FPS must be 0..8")
    args=["ffmpeg","-nostdin","-v","error","-i",str(video),"-vf",
          f"fps={fps},scale={sample_width}:{sample_height},format=gray",
          "-f","rawvideo","-pix_fmt","gray","-"]
    run=subprocess.run(args,check=True,capture_output=True)
    size=sample_width*sample_height
    if len(run.stdout)%size:
        raise RuntimeError("Diagnostic output frame is truncated")
    n=len(run.stdout)//size
    if n==0:
        raise ValueError("No diagnostic frames")
    if n>max_frames:
        raise ValueError("Too many frames: shorten clip or lower diagnostic FPS")
    data=np.frombuffer(run.stdout,dtype=np.uint8).reshape((n,sample_height,sample_width))
    avg=np.mean(data,axis=(1,2))
    motion=(np.abs(np.diff(data.astype(np.int16),axis=0)).mean(axis=(1,2))
             if n>1 else np.array([],dtype=np.float64))
    dark=int(np.count_nonzero(avg<12))
    white=int(np.count_nonzero(avg>247))
    still=int(np.count_nonzero(motion<1.0))
    notes=[]
    if dark/n>.24:
        notes.append("More than 24% nearly black sampled frames; review intentional fades")
    if white/n>.16:
        notes.append("More than 16% nearly white frames; possible blown-out flashes")
    if n>2 and still/max(1,n-1)>.86:
        notes.append("Little interframe motion; verify actual animated presentation")
    if np.percentile(avg,95)-np.percentile(avg,5)<8:
        notes.append("Very narrow scene brightness range, check visual contrast")
    return {"frames_sampled":n,"nearly_black":dark,"nearly_white":white,
            "still_adjacent_samples":still,
            "mean_brightness":round(float(avg.mean()),2),
            "mean_motion":round(float(motion.mean()),3) if n>1 else 0,
            "notes":notes,"automated_aesthetic_approval":False}

def report(video:Path)->dict:
    data=sample_video(video)
    data["video"]=video.name
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video",required=True,type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--regions-json", type=Path, help="Check every frame of one short shot")
    args=p.parse_args()
    if args.regions_json:
        from studio.shorts_factory import probe

        media = probe(args.video)
        stream = next(s for s in media["streams"] if s["codec_type"] == "video")
        fps = float(Fraction(stream["avg_frame_rate"]))
        regions = json.loads(args.regions_json.read_text(encoding="utf-8"))
        d = inspect_temporal(args.video, regions, stream["width"], stream["height"],
                             fps, int(stream["nb_frames"]))
    else:
        d=report(args.video)
    out=json.dumps(d,ensure_ascii=False,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(out,encoding="utf-8")
    print(out)
    if d.get("status") == "BLOCKED":
        raise SystemExit(2)

if __name__=="__main__":
    main()
