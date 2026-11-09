"""Voice-pinned mouth timing for sprite plates and continuous facial rigs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import subprocess
import unicodedata
from bisect import bisect_right
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
RHUBARB_SHAPES = frozenset("ABCDEFGHX")
REST = (0., 0., 0.)
SILENCE = {"", "sil", "sp", "<eps>"}
LABIAL_CLOSURE = {"b", "p", "m"}
VOWEL_POSES = {
    # Aperture, horizontal spread and rounding for Russian IPA.
    "a": (.92, .10, .0), "ɑ": (.92, .07, .0), "æ": (.78, .18, .0),
    "ɐ": (.66, .06, .0), "ə": (.39, .01, .0), "ʌ": (.59, .04, .0),
    "e": (.52, .18, .0), "ɛ": (.64, .18, .0), "ɪ": (.32, .15, .0),
    "i": (.27, .23, .0), "ɨ": (.29, .12, .0), "ɯ": (.29, .05, .0),
    "o": (.60, -.23, .78), "ɔ": (.70, -.19, .70), "ɵ": (.47, -.21, .70),
    "u": (.34, -.31, .95), "ʊ": (.38, -.24, .72), "ʉ": (.35, -.26, .83),
}


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


def _track_document(root: Path, reference: str) -> dict:
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise ValueError("Viseme track path must be relative")
    track = (root / reference).resolve()
    if root.resolve() not in track.parents or track.suffix.lower() != ".json" or not track.is_file():
        raise ValueError("Missing or out-of-project viseme track")
    if track.stat().st_size > 2_000_000:
        raise ValueError("Viseme track exceeds the safety limit")
    return json.loads(track.read_text(encoding="utf-8"))


def load_viseme_track(root: Path, reference: str, voice: Path) -> list[dict]:
    data = _track_document(root, reference)
    if (not isinstance(data, dict) or data.get("schema") != 1
            or data.get("engine") != "rhubarb_phonetic"
            or data.get("wav_sha256") != _wav_hash(voice)
            or not isinstance(data.get("visemes"), list)
            or not 1 <= len(data["visemes"]) <= 20001):
        raise ValueError("Viseme track must match the exact WAV")
    return data["visemes"]


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold().replace("ё", "е"))


def _phones(rows: list, duration: float) -> list[tuple[float, float, str]]:
    if not isinstance(rows, list) or not 1 <= len(rows) <= 20000:
        raise ValueError("Missing phoneme alignment")
    result = []
    cursor = 0.
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError("Expected a phoneme interval")
        start, end, phone = row
        if (not isinstance(phone, str) or len(phone) > 30 or
                any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end)) or
                not -1e-6 <= start < end <= duration + .015 or start < cursor - 1e-6
                or start >= duration):
            raise ValueError("Invalid or overlapping phoneme intervals")
        start = max(cursor, start)
        end = min(end, duration)
        if start > cursor:
            result.append((cursor, start, "sil"))
        phone = phone.strip()
        if phone[-2:] in {"_B", "_I", "_E", "_S"}:
            phone = phone[:-2]
        phone = "".join(c for c in unicodedata.normalize("NFD", phone.strip())
                        if not unicodedata.combining(c)).strip("ˈˌ0123456789").rstrip("ːʲʷ")
        if phone == "spn":
            raise ValueError("MFA marked an unaligned or out-of-vocabulary word")
        if end > start:
            result.append((start, end, phone))
        cursor = end
    if cursor < duration:
        result.append((cursor, duration, "sil"))
    if not any(p in VOWEL_POSES for _, _, p in result):
        raise ValueError("Phoneme alignment has no supported Russian vowels")
    return result


def import_mfa(document: dict, voice: Path, text: str) -> dict:
    """Validate Montreal Forced Aligner JSON from the exact dialogue WAV."""
    activity = speech_activity(voice)
    if activity["start"] is None:
        raise ValueError("No detectable speech in WAV")
    if not isinstance(document, dict):
        raise ValueError("Expected an MFA JSON document")
    try:
        tiers = document["tiers"]
        phones = tiers["phones"]["entries"]
        words = [word for _, _, word in tiers["words"]["entries"] if word not in SILENCE]
        duration = document["end"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MFA JSON requires words and phones tiers") from exc
    if (type(duration) not in (int, float) or not math.isfinite(duration) or
            abs(duration - activity["duration"]) > .015):
        raise ValueError("MFA timing has a different audio duration")
    expected = _words(text)
    if not expected or any(not isinstance(w, str) for w in words) or _words(" ".join(words)) != expected:
        raise ValueError("MFA words differ from the locked dialogue")
    aligned = _phones(phones, activity["duration"])
    spoken = [(a, b) for a, b, p in aligned if p not in SILENCE]
    if spoken[0][0] > activity["end"] or spoken[-1][1] < activity["start"]:
        raise ValueError("Phonemes do not overlap speech in the WAV")
    return {"schema": 2, "engine": "mfa_russian", "wav_sha256": _wav_hash(voice),
            "duration": activity["duration"], "phones": phones,
            "text_sha256": hashlib.sha256(" ".join(expected).encode()).hexdigest()}


def _phone_pose(phone: str) -> tuple[float, float, float]:
    if phone in VOWEL_POSES:
        return VOWEL_POSES[phone]
    if phone in SILENCE or phone in LABIAL_CLOSURE:
        return REST
    if phone in {"f", "v"}:
        return .10, .08, .0
    if phone in {"ʂ", "ʐ", "ɕ", "ʃ", "ʒ", "tɕ", "tʃ"}:
        return .22, -.11, .32
    return .22, .06, .0


def _ease(value: float) -> float:
    value = min(1., max(0., value))
    return value * value * (3. - 2. * value)


class PhonemeMotion:
    """Coarticulate neighboring phones and preserve a stable pose in real pauses."""

    def __init__(self, phones: list, activity: dict):
        self.duration = activity["duration"]
        self.phones = _phones(phones, self.duration)
        self.poses = [_phone_pose(phone) for _, _, phone in self.phones]
        self.transitions = []
        for (start, end, phone), (_, following_end, following) in zip(self.phones, self.phones[1:]):
            anticipation = min(.050, .45 * (end - start))
            release = min(.040, .45 * (following_end - end))
            # Pre-close the lips before labial stops, then hold the seal
            # through the aligned consonant, including short 30-fps intervals.
            left = 0. if phone in LABIAL_CLOSURE else anticipation
            right = 0. if following in LABIAL_CLOSURE else release
            self.transitions.append((end - left, end + right))
        self.starts = [start for start, _ in self.transitions]
        self.pauses = []
        start = None
        step = activity["step_ms"] / 1000
        for index, active in enumerate([*activity["active"], True]):
            if not active and start is None:
                start = index
            elif active and start is not None:
                if (index - start) * step >= .08:
                    self.pauses.append((start * step, index * step))
                start = None

    def pose(self, seconds: float) -> tuple[float, float, float]:
        if not 0 <= seconds < self.duration:
            return REST
        index = bisect_right(self.starts, seconds) - 1
        pose = self.poses[index + 1]
        if index >= 0:
            left, right = self.transitions[index]
            if seconds < right:
                weight = _ease((seconds - left) / (right - left))
                pose = tuple(a + (b - a) * weight
                             for a, b in zip(self.poses[index], pose))
        for start, end in self.pauses:
            if start - .025 < seconds < end + .025:
                weight = _ease(max(start - seconds, seconds - end) / .025)
                return tuple(value * weight for value in pose)
        return pose


def load_phoneme_motion(root: Path, reference: str, voice: Path, text: str) -> PhonemeMotion:
    data = _track_document(root, reference)
    text_hash = hashlib.sha256(" ".join(_words(text)).encode()).hexdigest()
    activity = speech_activity(voice)
    if (not isinstance(data, dict) or data.get("schema") != 2 or data.get("engine") != "mfa_russian"
            or data.get("wav_sha256") != _wav_hash(voice) or data.get("text_sha256") != text_hash
            or type(data.get("duration")) not in (int, float)
            or not math.isfinite(data["duration"])
            or abs(data["duration"] - activity["duration"]) > .015 or activity["start"] is None):
        raise ValueError("Phoneme track must match the exact WAV and locked dialogue")
    return PhonemeMotion(data.get("phones"), activity)


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
    engine = p.add_mutually_exclusive_group()
    engine.add_argument("--rhubarb", metavar="EXECUTABLE",
                        help="Generate voice-pinned visemes with local Rhubarb")
    engine.add_argument("--mfa-json", type=Path, help="Import Russian MFA JSON for a mouth mesh")
    p.add_argument("--text-file", type=Path, help="Locked dialogue used to produce MFA timing")
    args=p.parse_args()
    if args.mfa_json:
        if not args.text_file:
            p.error("--mfa-json requires --text-file")
        data = import_mfa(json.loads(args.mfa_json.read_text(encoding="utf-8")),
                          args.voice, args.text_file.read_text(encoding="utf-8"))
    elif args.rhubarb:
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
