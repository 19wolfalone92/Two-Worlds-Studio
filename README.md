# TWO WORLDS — Visual Production Studio

**Status:** tested technical motion-comic engine; professional illustrated master is **not yet produced**. Independently maintained from Williams Bot and Williams AI.

## Ownership
- **ChatGPT Work:** the only story/script/dialogue author. Once reviewed, Work delivers a locked episode JSON and scene descriptions.
- **This repository:** consistent-body artwork reference registry, compositing, shot direction, licensed recorded voice/audio, quality checks, review bundles and MP4 exports.
- **Human reviewer:** signs off designs, rights, screenplay, and publication. The studio never silently approves its own output.
- **No autonomous storywriting; no TikTok publication** in CI.

## Tested capabilities
- CPU-only original placeholder artwork and 2.5D compositing of production PNG layers (background, character, foreground), camera/parallax, haze and special effects.
- English text captions and optional **preview-only** robotic speech.
- Optional real ComfyUI image generation against user-operated GPU/approved model; reference-conditioned custom workflow needed for stable faces.
- H.264 vertical preview, high-quality 1080×1920/24fps production mastering when full approval requirements are satisfied.
- Validations: artwork quality flags, rights/sha, recurring physical-body identity versions and 3-angle references, voice actor version/rights/time sync, Work script lock fingerprint, basic video motion/black frames.

## Reproducible preview (Python 3.11+, FFmpeg)
```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m studio.visual_director --episode episodes/E01_THE_REFLECTION.json --output dist/shot_plan.json
python -m studio.art_director --episode episodes/E01_THE_REFLECTION.json --output-dir dist/art
python -m studio.visual_prep --demo --output-dir dist/premium_demo
python -m studio.production --episode episodes/E01_THE_REFLECTION.json --width 360 --height 640 --fps 10
```

E01/E02 are **old engineering fixtures**, not canonical Work scripts or final content. Placeholder images cannot be released using the production gate.

## Master release (requires actual paid/free GPU or human-made art)
```bash
python -m studio.work_contract --episode episodes/WORK_LOCKED_EPISODE.json
python -m studio.identity_qc --episode episodes/WORK_LOCKED_EPISODE.json --strict
python -m studio.audio_master --episode episodes/WORK_LOCKED_EPISODE.json --publish-gate
python -m studio.premium_assets --episode episodes/WORK_LOCKED_EPISODE.json --publish-gate
python -m studio.production --episode episodes/WORK_LOCKED_EPISODE.json --quality master --width 1080 --height 1920 --fps 24
```
The initial Work lock requires a real human approval, rights-clear images and voices, and cannot be issued by software itself. GPU inference is not available on default GitHub-hosted CPU runners. A command cannot make absent equipment or references exist.

## Relevant docs
- `docs/WORK_VISUAL_HANDOFF.md` — Work/Studio exchange contract, metadata and layer preparation.
- `docs/VISUAL_QUALITY_GATES.md` — release checklist.
- `docs/AI_STUDIO.md` — GPU art architecture and cost/hardware limits.
- `story/visual_bible.json`, `story/identity_registry.json` — physical-character appearance and reference versions (approval requires actual visual review).
- `audio/voice_registry.json` — speaker continuity and voice casting.
- `release_plan.json` — review-gated preparation schedule; no public posting.

## Status and workflow
The PR stack is staged as feature branches, not merged to main. Confirm passing CI and review before merge. **GitHub repo currently public** — switch visibility to Private if stories/art should remain undisclosed. Never commit credentials or copyrighted content without rights.
