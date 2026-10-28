# TWO WORLDS — Autonomous Motion Comic Studio (MVP)

Original international dark-fantasy / science-fiction comic series. **Independent** of Williams Bot and Williams AI.

## What really works

- Deterministic **free, offline** graphic-novel panels drawn programmatically (no PixVerse and no AI image API).
- Animated camera, rain, mirror resonance, particles, transitions and subtitles.
- Video and atmospheric soundtrack composed with Python, Pillow, NumPy and FFmpeg.
- Optional **offline robotic English narration** with `espeak-ng`.
- Production packaging: MP4 + timed SRT + caption/hashtags + QC manifest with SHA-256.
- Automated tests and scheduled GitHub Actions workflow. Scheduled work is **review-gated**, and generates an artifact only when an episode has both approvals and a UTC release date.

## What does NOT work automatically yet

- Photorealistic actors, hand-painted professional character art, automatic novel-quality scriptwriting, or a stable character-consistent image generator.
- Publishing to TikTok/Metricool from unattended GitHub Actions.
- Reading audience comments and choosing story branches.
- Guarantee of engagement, high production quality or commercial readiness.

The procedural art is an original *working visual stand-in*. Production illustrations can be replaced with legally usable images at `assets/` referenced from the episode JSON.

## Quick start (local)

Requirements: Python >=3.11, `ffmpeg` and `ffprobe` in PATH, optional `espeak-ng`.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m studio.production --episode episodes/E01_THE_REFLECTION.json --output-dir dist
# optional voice (requires espeak-ng):
python -m studio.production --episode episodes/E01_THE_REFLECTION.json --voiceover
```

Find `dist/E01/E01_preview.mp4`, `E01.srt`, `POSTING_PACKAGE.md` and `manifest.json`.

Use `--width 1080 --height 1920 --fps 24` for a higher-resolution render.

## Scheduled studio (no publishing)

See `release_plan.json`. To stage an episode on an intended UTC date:

1. Review complete script, art, narration, copyright/usage rights and consistency.
2. Set `release_date` to the intended UTC day, e.g. `2026-11-03`.
3. Explicitly set **both** `editorial_approved: true` and `rights_approved: true`.
4. Merge reviewed changes after passing CI.
5. The scheduled GitHub Actions run produces downloadable review artifacts. **It does not publish** or access Metricool credentials.

For testing a date manually:

```sh
python -m studio.production --scheduled --date 2026-11-03
```

Without approved content, nothing is rendered. GitHub cron is approximate and could run late or be skipped; a production scheduler would need missed-run handling and persistent storage.

## Governance

- All changes on feature branches and PR review first.
- Do not commit API keys, accounts, third-party media without rights, or private data.
- Publish only after explicit human approval and compliance checks.
- Check `docs/STORY_CANON.md` for hard identity and mirror invariants.
- This repository is **currently public**; change to **Private** under GitHub repository Settings if desired.

## Story continuity quality gate

`studio/continuity.py` verifies in every annotated panel whether the depicted
consciousness belongs in the corresponding Earth/Noctaris **body** at that point
of the timeline. It rejects impossible unsignaled transitions and consecutive
episodes with inconsistent start/end states. Approved scheduled episodes **must**
contain these annotations. This is a mechanized subset of continuity review,
not a substitute for a human screenplay editor.

## Art director and scriptwriter (experimental feature branch)

The new `studio.art_director` produces original detailed image prompts for
characters' **physical bodies** and settings. It persists stable body identity,
costume and reference hashes across episodes, and can send SDXL requests to a
user-run ComfyUI GPU server (only when explicitly enabled). This repository
does not include GPUs, checkpoints or paid image service credentials. Text
prompts and seeds alone do **not** ensure identical faces: supply rights-cleared
reference sheets and custom reference-conditioning (e.g. IP-Adapter + pose)
and manually inspect outputs.

```sh
python -m studio.art_director --episode episodes/E01_THE_REFLECTION.json --output-dir dist/art
# Real image generation requires running ComfyUI and an installed licensed checkpoint
python -m studio.art_director --backend comfyui --checkpoint your-model.safetensors
```

The new `studio.story_writer` uses locked story beats + persistent canon
snapshot and outputs **unapproved** episode drafts (E03, E04). Default offline
`outline` mode fills prepared beats; it is not an LLM novelist. Optional local
Ollama mode writes English scene text and dialogue constrained by the beats:
```sh
python -m studio.story_writer --episode E03 --engine outline
ollama pull qwen2.5:7b
python -m studio.story_writer --episode E03 --engine ollama --model qwen2.5:7b
```

Ollama must be running on your own machine or reachable server. Neither
writer output nor generated art is published or silently marked approved.
See `docs/AI_STUDIO.md`.
