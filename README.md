# TWO WORLDS — Visual Production Studio

## Developer quickstart — unified studio interface

Run `python -m pip install -e '.[validation]'`, then `python -m studio.cli doctor`.
Use `python -m studio.cli validate --kind episode --input episodes/E01_THE_REFLECTION.json` before generating. `python -m studio.cli voice-plan --speaker Mira --language en --text 'Hello.'` does **not** synthesize audio, and `python -m studio.cli kaggle-quota` does **not** contact Kaggle unless `--live` is given. Production and release remain subject to the existing explicit rights/quality gates.

Active developer guide: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md); document index: [docs/INDEX.md](docs/INDEX.md).

**Status:** tested technical motion-comic engine; professional illustrated master is **not yet produced**. Independently maintained from Williams Bot and Williams AI.

## Branch consolidation — 2026-11-09 (PR #51)

The validated Studio consolidation was **squash-merged into `main`** at
`31eb2ff915a834b419de2f5b9345921fa7acc17f`. GitHub CI on the integration
candidate completed four green checks: combined 391-test offline suite (13
skipped), Studio v2 CPU video/release contract, Studio v3 Kaggle API quota/
real-media verification, and ComfyUI reference-conditioning guard. Technical
preview export was also verified. The merge preserved the current v2/v3 gates.

`main` now includes the recovered character rig/art system, private-source
media packaging, Russian shorts production, music/sound assembly, accurate
frame-clock/subtitle/remaster helpers, private batch reconciliation, expanded
regression tests, manual Kaggle Russian shorts/character factory workflows, and
optional Elizabeth Bones voice preview scripts.

Excluded by design: **72 encrypted request payloads, 9 historic private
request/recipient files and 5 hard-coded old-branch production workflows**.
They remain in historical branches; no unpublished source media, credentials
or private decryption keys were copied to public `main`. Old branches must not
be pruned until independent private-artifact verification and archival.
**Passing CI does not prove a finished film, continuous character motion,
approved voice acting, real GPU capacity, or publication readiness.**

Full audit and remaining gates: `docs/STUDIO_CONSOLIDATION_AUDIT_20261109.md`.

## STORY GLITCH Studio v2 — production handoff (2026-11-06)

The existing engine now includes a **verified offline 3-part review assembler**
(`studio.triptych_batch`), strict vertical MP4 / first-three-second / cover
quality gate (`studio.release_gate`), safe Kaggle ComfyUI PNG receipt importer
(`studio.gpu_asset_handoff`), optional **Russian Qwen3-TTS**, CTC alignment of
actual speech, a Hugging Face read-only connection check, verified model cache
and actor-performance checks. The entire workflow preserves Work-authored
screenplays and stays free of mandatory paid services.

**See [docs/STUDIO_V2_STATUS.md](docs/STUDIO_V2_STATUS.md) for exact contracts,
commands, real status and limitations.** In particular, code-level CI success
is not an invented 24-frame ARCHIVE 17 art delivery or three finished films;
real approved media/voices still need to be staged privately. No CI job can
post unfinished videos.

## Ownership
- **ChatGPT Work:** the only story/script/dialogue author. Once reviewed, Work delivers a locked episode JSON and scene descriptions.
- **This repository:** consistent-body artwork reference registry, compositing, shot direction, licensed recorded voice/audio, quality checks, review bundles and MP4 exports.
- **Human reviewer:** signs off designs, rights, screenplay, and publication. The studio never silently approves its own output.
- **No autonomous storywriting; no TikTok publication** in CI.


## New: CPU hybrid STORY GLITCH mastering

`studio.hybrid_master` now assembles original 9:16 stills (slow camera motion) and optional pre-generated 3–5 s animation clips into one H.264/AAC vertical MP4. It aligns supplied per-shot voice tracks, mixes optional music, writes UTF-8 SRT and SHA-256/QA receipts, rejects overlong speech and requires explicit review flags for master exports. This editor runs **without GPU or API quotas**, but **does not generate imagery, motion, voices or scripts** itself and never publishes. Tested with real FFmpeg clips in `tests/test_hybrid_master.py`.

Usage, manifest JSON schema and honest resource limits: [docs/HYBRID_VIDEO_STUDIO.md](docs/HYBRID_VIDEO_STUDIO.md).

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
The validated shared production modules from PR #51 are merged to main. Older feature branches and their external private artifacts are retained for verified archival, not for automatic re-merging. **GitHub repo currently public** — switch visibility to Private if stories/art should remain undisclosed. Never commit credentials or copyrighted content without rights.

## Automated artwork import (Work → Studio)

Use the external image importer to stage approved **script content** plus
newly illustrated art layers (never scripts generated by this studio):

```bash
python -m studio.work_contract --episode episodes/WORK_EPISODE.json
python -m studio.visual_director --episode episodes/WORK_EPISODE.json --output dist/shots.json
python -m studio.asset_ingest --episode episodes/WORK_EPISODE.json \
  --image-map input/art_map.json --source-dir input/rendered_images \
  --assets assets --output dist/WORK_EPISODE_visual.json
python -m studio.production --episode-file dist/WORK_EPISODE_visual.json \
  --width 540 --height 960 --fps 12
```

The source script hash remains unchanged. Every imported layer is marked
**unreviewed**, not automatically approved; master render and publication
remain blocked until the visual, identity, voice and rights gates pass.
Optional `studio.lipsync` uses an artist-provided transparent mouth overlay
and the energy envelope of an approved recorded voice. It is **not** speech
recognition or full facial performance capture.

## Free model inference notebook (new)

A phone-operated **Kaggle notebook** is committed at
`notebooks/TWO_WORLDS_Kaggle_FLUX_Kokoro.ipynb`. It clones this feature branch
and has two explicit opt-in switches (both **False** by default):

- `ENABLE_GPU_IMAGE=True`: attempt **one real FLUX.2 klein 4B image** using the
  available Kaggle GPU, float16 on T4/P100 and CPU offloading. No real GPU
  generation has yet been tested. Kaggle free quota and memory availability vary.
- `ENABLE_KOKORO_VOICE=True`: synthesize a short English test WAV with
  Kokoro-82M. No voice has yet been rendered on Kaggle/approved.

The notebook creates an output ZIP to download from Kaggle. It never uploads,
posts, changes a screenplay, or approves its own materials. Work continues to
own **all scripts and dialogues**. Model and synth software licenses are
subject to their own original terms.

Full setup and acceptance checklist: `docs/FREE_KAGGLE_PRODUCTION.md`.

```bash
# Safe CPU-only preflight: no GPU/model download
python -m studio.kaggle_flux --mode character --body earth_mira --view front --dry-run
python -m studio.kokoro_tts --text "Voice test only." --speaker Mira --dry-run
```

## Kaggle API automation (manual; reviewed feature branch)

The studio now contains a GitHub Actions workflow that can call Kaggle's official API to **submit a private GPU kernel**. It uses the encrypted GitHub secret **KAGGLE_API_TOKEN** and a non-secret **KAGGLE_USERNAME** repository variable. A run must be started manually in Actions; no scheduled GPU or TikTok publication. The Kaggle worker packages source code into a kernel without transmitting GitHub credentials. The real portrait/voice files remain unreviewed.

**Critical:** The feature remains in a pull request; to see the GitHub Run workflow button, the workflow must be present on the default branch after review/merge. The API token must be added by the account owner in GitHub Settings; this chat cannot read it and nobody should paste it here. This public repository deliberately avoids uploading unpublished art to public Actions artifacts; download such results from the private Kaggle notebook. Details: docs/KAGGLE_API_AUTOMATION.md.

## Kaggle API: token-only bootstrap (account username auto-discovery)
The connected personal token was detected in GitHub Actions, but the
KAGGLE_USERNAME variable was empty in the initial connection check.
The manual Kaggle and connection-check workflows now attempt to determine
the authenticated Kaggle username via official CLI; no additional variable
is needed if introspection works. GPU usage still requires an explicit
manual workflow trigger, and the reference portraits remain unreviewed.

[Кинематографическая сборка, проверка MP4 и возобновление рендера](docs/CINEMATIC_PRODUCTION_RU.md)
