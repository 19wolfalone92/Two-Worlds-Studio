# Russian speech in TWO WORLDS Studio

**Scope:** a new voice-generation option for FUTURE Work-approved scripts. No old
stories, English narration, images, subtitles, music, scheduling, TikTok Studio
or Metricool are modified by this change.

## Engine

- `Qwen3-TTS` CustomVoice (0.6B default, 1.7B optional), Russian language.
- Open-source upstream: https://github.com/QwenLM/Qwen3-TTS
- Official models: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` and
  `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`; licensed Apache-2.0.
- Voices: Serena, Ryan, Vivian, Aiden, Sohee, Ono_Anna, Dylan, Eric, Uncle_Fu.
  None is guaranteed to sound native Russian; owner should audition.
- Styles: neutral, warm, sad, mysterious, tense, joyful. Style requests are
  instructions, not a guarantee of convincing emotional acting.
- Dedicated private GPU runtime recommended (Kaggle T4 possible, not validated
  with this backend). CPU loading may be slow. The model is fetched on first
  non-dry run; no weights are shipped in GitHub, and CI downloads nothing.

## Install once on your studio GPU runtime

Use a separate Python 3.12 environment:

```bash
python -m pip install qwen-tts soundfile numpy
```

Install the appropriate PyTorch/CUDA package for that GPU when not supplied by
the environment. The studio's standard requirements are unchanged.

## A single new Russian test voice (no existing story affected)

```bash
python -m studio.russian_qwen_tts \
  --text "Это тест русской озвучки нашей студии." \
  --voice Serena --style warm --output private/voice_test_ru.wav --dry-run
```

Remove `--dry-run` only in an actual GPU runtime with the model installed and
after reviewing the intended script.

## A new multi-story project (private local files only)

For new stories, put each original Russian text in a local `.txt` adjacent to
a local `jobs.json` file. Do not upload unapproved scripts to this public repo.

```json
[
  {
    "id": "new_story_01",
    "text_file": "my_new_story_ru.txt",
    "language": "ru",
    "voice": "qwen:Serena",
    "style": "sad"
  },
  {
    "id": "new_story_02",
    "text_file": "another_new_story_ru.txt",
    "language": "ru",
    "voice": "qwen:Ryan",
    "style": "mysterious"
  }
]
```

```bash
python -m studio.narration_batch --manifest private/jobs.json \
  --output-dir private/voices --dry-run
# Remove --dry-run only when GPU dependencies are installed and paths are approved.
```

The same batch interface keeps English Kokoro presets `af_heart`, `bm_george`,
etc. (`language: "a"` / `"b"`) unchanged. Existing English-only batch
manifest filenames are preserved.

## Safety / acceptance criteria

- No story rewrite, automatic translation, voice cloning, publishing or merges.
- Russian text must contain Cyrillic; English voice IDs cannot select Russian.
- Input filenames cannot escape the private manifest directory via `..` or symlink.
- Dry run does not require a model or produce WAV files.
- Real inference validates rate, sample data, duration and WAV headers, refuses
  overwrites, logs SHA-256, and marks recordings `unreviewed`.
- Narration is NOT approved for final production until a human checks naturalness,
  pronunciation, coherence, emotionality, audio rights and timing.
- No real Qwen3 GPU inference has been demonstrated in this PR, only code and
  offline contract tests. Actual synthesized Russian voice is a separate QA gate.
