"""Original scene underscore from a supplied public-domain piano SoundFont.

No sampled songs, external service, or model licence ambiguity. Compositions
remain subject to artistic listening review; this module never approves a video.
"""
from __future__ import annotations
import hashlib
import math
import os
import struct
import tempfile
import wave
from pathlib import Path
import numpy as np


class Piano:
    def __init__(self, path: Path):
        data = Path(path).read_bytes()
        if data[:4] != b'RIFF' or data[8:12] != b'sfbk':
            raise ValueError('Expected a real SoundFont2 piano file')
        chunks = {}
        def parse(start, end):
            pos = start
            while pos+8 <= end:
                tag = data[pos:pos+4]
                size = struct.unpack_from('<I', data, pos+4)[0]
                if pos+8+size > len(data):
                    raise ValueError('Damaged SoundFont chunk')
                if tag in (b'RIFF', b'LIST'):
                    parse(pos+12, pos+8+size)
                else:
                    chunks[tag] = data[pos+8:pos+8+size]
                pos += 8+size+(size%2)
        parse(0, len(data))
        self.pcm = np.frombuffer(chunks[b'smpl'], '<i2').astype(np.float32)/32768
        headers = chunks[b'shdr']
        self.samples = []
        for off in range(0, len(headers)-46, 46):
            name, start, end, loop_start, loop_end, rate, root, cents, link, kind = struct.unpack_from(
                '<20sIIIIIBbHH', headers, off)
            if not 0 <= start < loop_start < loop_end <= end <= len(self.pcm):
                raise ValueError('Invalid piano sample loop')
            self.samples.append((start, end, loop_start, loop_end, rate, root, cents))
        if not self.samples:
            raise ValueError('Piano samples are missing')
        self.source_sha256 = hashlib.sha256(data).hexdigest()

    def note(self, midi: int, held: float, sample_rate: int, velocity: float) -> np.ndarray:
        start, end, ls, le, native, root, cents = min(self.samples, key=lambda x: abs(x[5]-midi))
        seconds = min(8.0, held+1.25)
        t = np.arange(round(seconds*sample_rate), dtype=np.float64)/sample_rate
        position = start+t*native*2**((midi-root-cents/100)/12)
        after = position >= le
        position[after] = ls+(position[after]-ls)%(le-ls)
        audio = np.interp(position, np.arange(start, end), self.pcm[start:end])
        envelope = np.minimum(t/.008, 1)*np.exp(-t/(1.5+held*.45))
        envelope *= np.where(t <= held, 1, np.exp(-(t-held)*4.2))
        return (audio*envelope*velocity).astype(np.float32)


PROFILES = {
    'sad': (52, 62, [(0, 3, 7, 14), (8, 12, 15, 19), (5, 8, 12, 19), (7, 10, 14, 17)], .045),
    'tender': (53, 67, [(0, 4, 7, 14), (9, 12, 16, 19), (5, 9, 12, 16), (7, 11, 14, 21)], .052),
    'mysterious': (50, 58, [(0, 7, 14, 15), (3, 10, 14, 19), (8, 12, 19, 22), (7, 14, 17, 22)], .034),
    'tense': (52, 72, [(0, 7, 13, 15), (1, 8, 12, 19), (5, 12, 15, 20), (7, 10, 13, 19)], .032),
    'hopeful': (55, 66, [(0, 4, 7, 14), (5, 9, 12, 19), (9, 12, 16, 19), (7, 11, 14, 21)], .050),
    'neutral': (50, 60, [(0, 7, 12, 16), (5, 12, 16, 19), (9, 12, 16, 23), (7, 14, 19, 21)], .039),
}


def compose_score(dest: Path, seconds: float, piano: Piano, *, mood='sad', seed=0, sample_rate=44100):
    if mood not in PROFILES or not .1 <= seconds <= 600:
        raise ValueError('Invalid scene music duration or mood')
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError('Preserve the existing scene score')
    root, bpm, progression, level = PROFILES[mood]
    rng = np.random.default_rng(seed)
    transpose = int(rng.choice([-3, -2, 0, 2, 3]))
    root += transpose
    beat = 60/bpm
    length = round(seconds*sample_rate)
    signal = np.zeros((length, 2), dtype=np.float32)
    def add_note(note, start, held, strength):
        if start >= seconds:
            return
        audio = piano.note(note, held, sample_rate, strength)
        offset = max(0, round(start*sample_rate))
        count = min(len(audio), length-offset)
        pan = max(.18, min(.82, .5+(note-66)*.01+rng.normal(0, .025)))
        signal[offset:offset+count, 0] += audio[:count]*math.sqrt(1-pan)
        signal[offset:offset+count, 1] += audio[:count]*math.sqrt(pan)
    bar = 0
    while bar*4*beat < seconds:
        start = bar*4*beat
        chord = progression[(bar+seed)%len(progression)]
        density = .45 if mood in {'mysterious', 'tense'} else .76
        swell = .65+.35*math.sin(math.pi*min(start/max(seconds,1),1))
        add_note(root+chord[0]-12, start+.02, beat*3.5, .35*swell)
        for k in range(8):
            if rng.random() > density:
                continue
            interval = chord[(k+bar)%len(chord)]
            note = root+interval+(12 if k in [3, 7] and mood!='tense' else 0)
            add_note(note, start+k*beat/2+rng.uniform(-.025,.025),
                     beat*rng.uniform(.8,1.8), rng.uniform(.26,.48)*swell)
        bar += 1
    # Sparse stereo early reflections, never a competing lead melody.
    for delay, attenuation in [(0.089,.13),(.173,.10),(.293,.075),(.431,.045)]:
        frames = round(delay*sample_rate)
        if frames < length:
            signal[frames:, 0] += signal[:-frames, 1]*attenuation
            signal[frames:, 1] += signal[:-frames, 0]*attenuation
    rms = float(np.sqrt(np.mean(signal**2)))
    if rms <= 1e-8:
        raise ValueError('Composed piano score is silent')
    signal *= level/rms
    t = np.arange(length)/sample_rate
    fade = np.minimum(1,np.minimum(t/.65, (seconds-t)/1.3)).clip(0,1)
    signal *= fade[:, None]
    peak = float(np.max(np.abs(signal)))
    if peak > .7:
        signal *= .7/peak
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix='.wav')
    os.close(fd)
    try:
        with wave.open(tmp, 'wb') as out:
            out.setnchannels(2);out.setsampwidth(2);out.setframerate(sample_rate)
            out.writeframes((signal*32767).astype('<i2').tobytes())
        os.link(tmp, dest)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return {'path': str(dest), 'seconds': length/sample_rate, 'source': 'original_composition',
            'instrument': 'Florestan piano SoundFont, public-domain samples',
            'sample_source_sha256': piano.source_sha256, 'mood': mood, 'seed': seed,
            'artistic_listening_review': 'pending'}
