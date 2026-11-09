"""Waveform-gated 2D lipsync with optional local Rhubarb mouth timing."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
RHUBARB_SHAPES = frozenset("ABCDEFGHX")


def _voice_rms(path: Path, step_ms: int) -> tuple[np.ndarray, float]:
    if not (20<=step_ms<=100):
        raise ValueError("Envelope step must be 20..100ms")
    output=subprocess.run(["ffmpeg","-nostdin","-v","error",
            "-i",str(path),"-map","0:a:0","-vn","-ac","1","-ar",str(SAMPLE_RATE),
            "-t","601","-f","s16le","-"],capture_output=True,check=True,timeout=60).stdout
    if len(output)>SAMPLE_RATE*2*600:
        raise ValueError("Voice sample exceeds safety maximum")
    if len(output)<640:
        raise ValueError("Voice sample too short")
    signal=np.frombuffer(output,dtype="<i2").astype(np.float32)/32768
    chunk=round(SAMPLE_RATE*step_ms/1000)
    level=np.asarray([float(np.sqrt(np.mean(signal[i:i+chunk]**2)))
                      for i in range(0,len(signal),chunk)],dtype=np.float32)
    return level, len(signal) / SAMPLE_RATE


def speech_activity(path: Path, step_ms: int = 20) -> dict:
    """Do not smear speech energy into pauses. A quiet tail closes the mouth."""
    level, duration = _voice_rms(path, step_ms)
    floor = max(.0012, float(np.percentile(level, 95)) * .025)
    active = (level >= floor).tolist()
    indices = [i for i, value in enumerate(active) if value]
    return {"step_ms": step_ms, "active": active, "duration": duration,
            "start": indices[0] * step_ms / 1000 if indices else None,
            "end": min(duration, (indices[-1] + 1) * step_ms / 1000) if indices else None}


def voice_envelope(path:Path, step_ms=50)->np.ndarray:
    level, _ = _voice_rms(path, step_ms)
    # First-order smoothing removes distracting mechanical mouth chattering.
    if len(level)>2:
        level=np.convolve(level,[.2,.6,.2],mode="same").astype(np.float32)
    peak=float(np.percentile(level,95)) if len(level) else 0.0
    if peak<=1e-5:
        return np.zeros_like(level)
    return np.clip(level/peak,0,1)


def _wav_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rhubarb_visemes(document: dict, duration: float) -> list[dict]:
    """Convert Rhubarb intervals to ordered shape transitions, retaining gaps."""
    rows = document.get("mouthCues") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not 1 <= len(rows) <= 20000:
        raise ValueError("Missing Rhubarb mouth timing")
    changes = {}
    previous_end = 0.
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid mouth interval")
        start, end, value = row.get("start"), row.get("end"), row.get("value")
        if (any(type(n) not in (int, float) or not math.isfinite(n) for n in (start, end))
                or not previous_end <= start < end <= duration + .015
                or start >= duration or not isinstance(value, str) or value not in RHUBARB_SHAPES):
            raise ValueError("Overlapping or invalid Rhubarb timing")
        changes[float(start)] = "rest" if value == "X" else value
        changes[min(duration, float(end))] = "rest"
        previous_end = end
    return [{"at": at, "shape": shape} for at, shape in sorted(changes.items())]


def recognize(path: Path, executable: str = "rhubarb") -> dict:
    if path.suffix.lower() != ".wav":
        raise ValueError("Rhubarb lip-sync input must be a WAV")
    activity = speech_activity(path)
    if activity["start"] is None:
        raise ValueError("No detectable speech in WAV")
    result = subprocess.run(
        [executable, "--recognizer", "phonetic", "--exportFormat", "json",
         "--threads", "2", str(path.resolve())],
        capture_output=True, text=True, check=True, timeout=120,
    )
    document = json.loads(result.stdout)
    cues = rhubarb_visemes(document, activity["duration"])
    if not any(cue["shape"] not in {"rest", "A"} for cue in cues):
        raise ValueError("Recognizer produced no usable mouth shapes")
    return {"schema": 1, "engine": "rhubarb_phonetic",
            "wav_sha256": _wav_hash(path), "duration": activity["duration"],
            "speech_start": activity["start"], "speech_end": activity["end"],
            "visemes": cues, "requires_listening_and_visual_review": True}


def load_viseme_track(root: Path, reference: str, voice: Path) -> list[dict]:
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise ValueError("Viseme track path must be relative")
    track = (root / reference).resolve()
    if root.resolve() not in track.parents or track.suffix.lower() != ".json" or not track.is_file():
        raise ValueError("Missing or out-of-project viseme track")
    if track.stat().st_size > 2_000_000:
        raise ValueError("Viseme track exceeds the safety limit")
    data = json.loads(track.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("schema") != 1
            or data.get("engine") != "rhubarb_phonetic"
            or data.get("wav_sha256") != _wav_hash(voice)
            or not isinstance(data.get("visemes"), list)
            or not 1 <= len(data["visemes"]) <= 20001):
        raise ValueError("Viseme track must match the exact WAV")
    return data["visemes"]

def mouth_open(envelope:np.ndarray,scene_seconds:float,*,step_ms=50,
               threshold=.22,delay_ms=0)->bool:
    if not len(envelope):
        return False
    if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not .05<=threshold<=.8:
        raise ValueError("Mouth threshold outside usable range")
    index=math.floor((scene_seconds*1000-delay_ms)/step_ms)
    if index<0 or index>=len(envelope):
        return False
    return bool(envelope[index]>=threshold)

def main():
    p=argparse.ArgumentParser(description="Inspect sound-driven mouth motion cues")
    p.add_argument("--voice",type=Path,required=True)
    p.add_argument("--output",type=Path)
    p.add_argument("--rhubarb", metavar="EXECUTABLE",
                   help="Generate voice-pinned visemes with an installed local Rhubarb binary")
    args=p.parse_args()
    if args.rhubarb:
        data = recognize(args.voice, args.rhubarb)
    else:
        data = {"voice": str(args.voice), "step_ms": 50,
                "levels": voice_envelope(args.voice).tolist()}
    content=json.dumps(data,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(content,encoding="utf-8")
    print(content)

if __name__=="__main__":
    main()
