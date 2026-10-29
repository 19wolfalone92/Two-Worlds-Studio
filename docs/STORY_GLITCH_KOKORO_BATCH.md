# Reusing the TWO WORLDS Studio Kokoro engine for external stories

`studio.narration_batch` accepts user-authored English plain-text stories from a **local private folder** and calls the already-existing `studio.kokoro_tts.synthesize` with official Kokoro-82M voice presets.

This is an **engine integration**, not a replacement for the story-writing workflow. The Studio does not write scripts, invent rights-clear materials or publish.

## Input
Provide a JSON array in `stories.json` alongside the private `.txt` files:

```json
[
  {"id": "story01", "text_file": "story01.txt", "voice": "bm_george", "language": "b"},
  {"id": "story02", "text_file": "story02.txt", "voice": "af_heart", "language": "a"}
]
```

Run validation (does **not** generate audio):

```bash
python -m studio.narration_batch --manifest /private/stories.json --output-dir /private/output --dry-run
```

When the actual Kokoro model, weights, `kokoro>=0.9.2`, `soundfile` and `espeak-ng` are available:

```bash
python -m studio.narration_batch --manifest /private/stories.json --output-dir /private/output
```

The engine emits one WAV per story plus `kokoro_batch_manifest.json` with file hashes; voices remain unreviewed. SRT timing and full MP4 audio mixing are separate presentation steps and are not certified by this stage.

## Limits

- A GitHub-hosted CPU CI success does not imply Kokoro audio was generated; tests are offline dry-run checks.
- First inference requires downloading model weights from an approved public source, e.g. a network-enabled private Kaggle Notebook with enough CPU/GPU resources.
- Do not commit original Story Glitch scripts, private images, unreleased WAVs, or Kaggle secrets into the public TWO WORLDS repository.
- Final release needs a listening review and permission checks. The studio's Two Worlds continuity and Work-only screenplay rules remain unchanged.
