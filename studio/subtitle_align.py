"""Local Russian CTC alignment of the actual recorded waveform to locked text.

No invented word timings: unsupported text or an impossible alignment raises.
The optional speech model must already be staged on disk by the GPU worker.
"""
from __future__ import annotations
import json
import re
import wave
from pathlib import Path
import numpy as np

MODEL = 'jonatasgrosman/wav2vec2-large-xlsr-53-russian'


def ctc_spans(logp: np.ndarray, labels: list[int], blank: int) -> list[tuple[int, int, float]]:
    """Viterbi over the complete CTC graph, including repeated-letter blanks."""
    if logp.ndim != 2 or not labels or not np.isfinite(logp).all():
        raise ValueError('Missing finite CTC emissions or source letters')
    states = np.full(len(labels)*2+1, blank, dtype=np.int64)
    states[1::2] = labels
    allowed_skip = np.zeros(len(states), dtype=bool)
    allowed_skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    prev = np.full(len(states), -np.inf, dtype=np.float64)
    prev[0] = 0
    directions = np.zeros((len(logp), len(states)), dtype=np.uint8)
    for t, emission in enumerate(logp):
        one = np.r_[-np.inf, prev[:-1]]
        two = np.r_[-np.inf, -np.inf, prev[:-2]]
        two[~allowed_skip] = -np.inf
        choices = np.stack([prev, one, two])
        direction = choices.argmax(axis=0)
        directions[t] = direction
        prev = choices[direction, np.arange(len(states))] + emission[states]
    final = len(states)-1 if prev[-1] > prev[-2] else len(states)-2
    if not np.isfinite(prev[final]):
        raise ValueError('Waveform is too short for the original transcript')
    frames = [[] for _ in labels]
    for t in range(len(logp)-1, -1, -1):
        if final % 2:
            frames[(final-1)//2].append(t)
        final -= int(directions[t, final])
    if any(not f for f in frames):
        raise ValueError('Original source letters could not all be aligned')
    return [(min(f), max(f)+1, float(np.mean(logp[f, labels[i]])))
            for i, f in enumerate(frames)]


def readable_cues(words: list[dict], max_chars: int = 70) -> list[dict]:
    cues, current = [], []
    def finish():
        if current:
            cues.append({'start': current[0]['start'], 'end': current[-1]['end'],
                         'text': ' '.join(w['word'] for w in current)})
            current.clear()
    for word in words:
        if current and (len(' '.join(w['word'] for w in current) + ' ' + word['word']) > max_chars
                        or word['end'] - current[0]['start'] > 5.5):
            finish()
        current.append(word)
        if len(current) >= 3 and re.search(r'[.!?…][»"]?$', word['word']):
            finish()
    finish()
    return cues


def align_voice(audio_path: Path, measured_cues: list[dict], model_dir: Path,
                *, device: str = 'cuda', processor=None, model=None) -> dict:
    import torch
    if processor is None or model is None:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        processor = Wav2Vec2Processor.from_pretrained(str(model_dir), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    with wave.open(str(audio_path), 'rb') as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError('CTC requires mono 16-bit voice source')
        rate = wav.getframerate()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype(np.float32)/32768
    vocab = processor.tokenizer.get_vocab()
    lower = any(c in vocab for c in 'абв')
    blank = processor.tokenizer.pad_token_id
    all_words = []
    for cue in measured_cues:
        original = cue['text'].split()
        normalized = []
        for word in original:
            text = re.sub(r'[^А-Яа-яЁёA-Za-z]', '', word)
            text = text.lower() if lower else text.upper()
            if text and any(c not in vocab for c in text):
                raise ValueError('Source contains letters unsupported by the staged Russian aligner')
            normalized.append(text)
        labels, owners = [], []
        for i, text in enumerate(normalized):
            if not text:
                continue
            if labels:
                labels.append(vocab['|']); owners.append(None)
            for char in text:
                labels.append(vocab[char]); owners.append(i)
        part = pcm[round(cue['start']*rate):round(cue['end']*rate)]
        if len(part) < 100:
            raise ValueError('Measured voice cue contains no waveform')
        sample = np.interp(np.arange(round(len(part)*16000/rate))*rate/16000,
                           np.arange(len(part)), part).astype(np.float32)
        values = processor(sample, sampling_rate=16000, return_tensors='pt').input_values.to(device)
        with torch.inference_mode():
            emissions = model(values).logits[0].log_softmax(-1).float().cpu().numpy()
        spans = ctc_spans(emissions, labels, blank)
        frame_seconds = len(part)/rate/len(emissions)
        for i, word in enumerate(original):
            matched = [s for s, owner in zip(spans, owners) if owner == i]
            if matched:
                start = cue['start'] + matched[0][0]*frame_seconds
                end = cue['start'] + matched[-1][1]*frame_seconds
                confidence = float(np.exp(np.mean([s[2] for s in matched])))
                all_words.append({'word': word, 'start': round(start, 3),
                                  'end': round(end, 3), 'confidence': round(confidence, 5)})
            elif all_words:
                # Standalone punctuation belongs to its neighboring phrase;
                # it has no spoken duration and never receives fake timing.
                all_words[-1]['word'] += ' ' + word
            else:
                raise ValueError('Leading source punctuation cannot be assigned a word interval')
    expected = ' '.join(c['text'].strip() for c in measured_cues)
    if ' '.join(w['word'] for w in all_words) != expected:
        raise ValueError('Alignment changed or omitted original source words')
    return {'model': MODEL, 'alignment': 'ctc_forced_alignment_of_actual_waveform',
            'words': all_words, 'subtitles': readable_cues(all_words),
            'all_source_words_preserved': True,
            'low_confidence_words': sum(w['confidence'] < .03 for w in all_words)}
