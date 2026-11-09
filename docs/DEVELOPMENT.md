# TWO WORLDS STUDIO — Development guide (v4 foundation)

This is the **developer entrypoint**, not a claim that generated films or GPU
workflows have passed a live artistic review. The current GitHub repository is
public; unpublished scripts, scene artwork, voice WAVs, private input images,
secrets and unencrypted production datasets must stay **outside** this repo.

## First run (Python 3.11+)

Install Python and FFmpeg/ffprobe locally; espeak-ng only for older preview
speech. Start with CPU-only dependencies:

    python -m pip install -e '.[validation]'
    python -m studio.cli doctor
    python -m studio.cli validate --kind episode --input episodes/E01_THE_REFLECTION.json
    python -m studio.cli validate --kind visual_bible --input story/visual_bible.json
    python -m studio.cli validate --kind identities --input story/identity_registry.json
    python -m unittest discover -s tests -v

The older \`requirements.txt\`, \`requirements-cinematic.txt\`, and
\`requirements-private-output.txt\` are retained for deployed notebooks and
GitHub Actions. \`pyproject.toml\` now specifies canonical Python package
metadata with optional extras:

- \`.[validation]\`: typed core preflight, Pydantic 2; no model weights.
- \`.[cinematic]\`: OpenCV support for optional advanced animation.
- \`.[private]\`: cryptography for encrypted private artifact transport.
- \`.[voice]\`: SoundFile helper; actual Kokoro/Qwen model wheels and model
  licenses are resolved on the designated private worker, not at import.
- \`.[dev]\`: validation and encrypted transport test dependencies.

This is **not yet a universal lockfile** across Kaggle's GPU images and local
CPUs. Do not delete old requirements until every target environment has been
migrated and tested reproducibly.

## Single command surface

Use the console command \`two-worlds\` or \`python -m studio.cli\`.

| Command | Effect | Network / GPU |
| --- | --- | --- |
| \`doctor\` | Installed binaries and configured-secret booleans only. No secret values. | None |
| \`validate --kind episode --input FILE\` | Typed episode schema validation. | None |
| \`validate --kind visual_bible --input FILE\` | Visual bible contract. | None |
| \`validate --kind identities --input FILE\` | Physical body references schema. | None |
| \`voice-plan --speaker Mira --language en --text "Hello."\` | Returns voice route and text SHA, not the private text. | None |
| \`voice-plan --speaker Elizabeth --language ru --text "Привет."\` | Qwen route and speaker identity; does not synthesize. | None |
| \`quality --video FILE --srt FILE --cover FILE\` | Read-only technical film gate; no rights approval. | Local FFmpeg only |
| \`kaggle-quota\` | Offline dynamic quota warning. | None |
| \`kaggle-quota --live\` | Exactly one Kaggle CLI quota attempt; **compute hours are not API rate limits**. | One read-only API query |
| \`run --engine qwen\` | Only returns a plan for the Qwen entrypoint, without arguments or text. | None |

To invoke an existing external model or API tool, use:

    python -m studio.cli run --engine qwen --execute --allow-provider-usage -- --text "..." --voice Serena --output /private/voice.wav

For model invocation, provider usage is **explicit**, never inferred from a
dry-run. This wrapper does not magically prevent external provider charges or
verify independent provider bills; check the provider's free quota/licensing
and choose a private output destination before approving. The underlying
backend validates its own arguments and is still the authority. Do not paste
tokens into command lines, filenames, GitHub commits, or logs.

For built-in voice routing (also guarded):

    python -m studio.cli voice-plan --speaker Mira --language en --text "Hello."
    python -m studio.cli voice-plan --speaker Kostya --language ru --text "Привет." --execute --allow-model-load --output /private/kostya.wav

The \`--execute\` example requires installed voice models and explicit user
consent to run them. Generated audio remains **unreviewed**.

## Source-of-truth module boundaries

1. **Scripts:** Work-produced, locked JSON with authorship and SHA in private
   storage; no engine silently writes or translates approved dialogue.
2. **Contracts:** \`studio/contracts.py\` validates typed episode, visual
   bible, identity registry. It permits extra Work-defined metadata for
   forward compatibility, but checks required core fields. Use
   \`validate_bundle()\` to check scene location / physical body references.
3. **GPU providers:** \`studio/kaggle_api.py\`, \`studio/kaggle_api_limits.py\`,
   \`studio/comfy_kaggle.py\`, \`studio/gpu_asset_handoff.py\`,
   \`studio/cloudflare_flux.py\`, \`studio/free_studio.py\`. A token's presence
   is not proof of quota, approved art or working generation.
4. **Queue:** \`studio/production_queue.py\` is a private SQLite CPU queue,
   keyed by hashes, with bounded retry. It is **not** a running cloud worker.
   Never retry Kaggle 429 in a hot loop or assume a hidden fixed rate limit.
5. **Voice:** \`studio/voice_engine.py\` implements one preflight/dispatch
   contract. Approved identities belong to \`audio/voice_registry.json\`,
   Kokoro English casting in \`audio/kokoro_cast.json\`, Qwen Russian casting
   in \`audio/voice_routes.json\`. Existing synthesis backends remain
   \`studio/kokoro_tts.py\` and \`studio/russian_qwen_tts.py\`.
6. **Editing:** \`studio/hybrid_master.py\`, \`studio/triptych_batch.py\`
   and \`studio/shorts_factory.py\`; do not substitute a placeholder for
   missing real source frames or recorded/generated approved voice.
7. **Release:** \`studio/quality_hub.py\` consolidates technical film
   \`studio/release_gate.py\`, physical-body identity
   \`studio/identity_qc.py\`, and independent transcript auditing from
   \`studio/voice_audit.py\`. A \`technical_pass_human_review_required\`
   is *not* publication authorization.

## Tests and QA matrix

A lightweight developer workflow is
\`.github/workflows/studio-architecture-v4.yml\`. It installs the validation
extra and exercises typed contracts, voice registry planning, non-execution
guards and read-only quality reports, all with offline mocks.

The existing \`.github/workflows/ci.yml\` must also pass. It renders and
probes real H.264/AAC engineering proof clips, not film-ready final images.
The existing v2/V3 and ComfyUI reference graph contract checks stay in place.

Before promoting a 1080x1920/30 FPS episode, independently inspect actual
opening frames, character continuity, costumes, anatomy, subtitle alignment,
music loudness, narrator pronunciation, visible story clues and rights.
\`quality\` may combine these *technical* signals, but it cannot replace
human subjective artistic review. No TikTok/Metricool publication from CI.

## Documentation location

Read \`README.md\` for entrypoint; this page for active development. Historical
batch-specific details stay in \`docs/\` to avoid losing audit provenance. See
\`docs/INDEX.md\` to separate active contracts from archived notes. The
consolidation source log remains
\`docs/STUDIO_CONSOLIDATION_AUDIT_20261109.md\` and is not to be rewritten
or discarded without retention review.

## Known limitations / next work

- Migrate each branch-specific Kaggle worker to canonical entrypoint and
  dynamic quota supervisor **after** separate live compatibility tests.
- Implement exact reproducible CPU/GPU lockfiles and pinned model revisions
  once each target environment has been proven (Poetry/Hatch alone will not
  ensure CUDA wheel compatibility).
- Enforce typed preflight in every legacy GPU producer after fixtures have
  been migrated without breaking existing live jobs.
- Extend the review hub with opt-in captured encoded audio loudness and
  scene motion continuity; keep the current release gate fail-closed.
- Keep the public repo clean of private content and do not prune old
  production branches until encrypted handoffs are recovered.
