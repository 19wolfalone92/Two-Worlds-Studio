# STORY GLITCH / TWO WORLDS — Studio v3 provider foundation

**Status:** foundation implemented and tested offline; NOT 3 finished videos.
No existing Work screenplay, published episode, or Williams Bot was changed.

## Why this change

The connected account has verified credentials for Cloudflare Workers AI,
OpenRouter, Gemini, Hugging Face and AI Horde. Groq registration is blocked.
Kaggle output readback previously hit HTTP 429. Presence of keys is *not*
proof of free quota, generation quality, availability, or image production.

Existing `studio.hybrid_master`, `studio.triptych_batch` and continuity
and reference checks remain authoritative release paths.

## New working modules

- `studio/free_studio.py` — offline provider inventory and scene routing. Uses
  environment variable presence only; never reads secrets into output. Strictly
  distinguishes **configured**, **authenticated** and **inference verified**.
  Defaults to NO external calls and NO automatic uploads/publishing.
- `studio/cloudflare_flux.py` — real, **explicitly opt-in** single-frame
  Cloudflare FLUX.2 [klein] 4B image API transport. POST multipart/form-data,
  full-size image prompt + up to four SMALL input images (<512x512 each),
  deterministic seed, PNG pixel/size validation, SHA256 provenance, private
  local unreviewed review receipt. Zero automatic bulk or fallback requests.
- `studio/free_connectivity.py` — read-only Cloudflare account/model check,
  ensuring Account ID and token can actually list the model, not merely verify
  a generic token. This does NOT spend generation quota.
- `studio/production_queue.py` — PRIVATE persistent SQLite CPU queue:
  idempotent scene hashes, exclusive worker leases, retriable HTTP429/quota
  errors (with bounded exponential backoff and daily quota cool-down), no raw
  prompt/credentials/error logging. Jobs can be staged offline from plans.
  This is a queue **foundation**, not an active distributed worker.
- `tests/test_studio_v3_free.py`, `tests/test_studio_v3_connectivity.py`,
  `tests/test_production_queue.py` — fake HTTP, reference payload, API
  errors, queue restart, leases, image outputs, privacy and idempotency tests.
- `.github/workflows/studio-v3-free-tests.yml` — PR offline CI plus
  manual metadata-only Cloudflare preflight.

## GitHub Actions configuration

Already detected:
- Secret `KAGGLE_API_TOKEN` — Kaggle, current readback not verified.
- Secret `HF_TOKEN` — Hugging Face metadata verified.
- Secret `CLOUDFLARE_AI_TOKEN` — token checked for Cloudflare.
- Secret `GEMINI_API_KEY` — token authentication checked; no AI inference.
- Secret `OPENROUTER_API_KEY` — token authentication checked; no AI inference.
- Secret `HORDE_API_KEY` — token authentication checked; voluntary external hosts.
- **Variable** `CLOUDFLARE_ACCOUNT_ID` — newly configured.

Not present / OPTIONAL:
- `GROQ_API_KEY` — signup blocked; **must not block production**.
- `AZURE_SPEECH_KEY` + variable `AZURE_SPEECH_REGION`.
- `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`,
  variable `R2_BUCKET_NAME` — **do not require paid R2 billing**.
- variable `CLOUDFLARE_D1_DATABASE_ID`, variable
  `CLOUDFLARE_QUEUE_NAME` — not yet provisioned.
- Modal keys and Azure/Oracle/Google Cloud billing are not needed and have
  **no auto-enabled billing path**.

## Usage (offline safety check)

```bash
python -m pip install -r requirements.txt
python -m unittest tests.test_studio_v3_free tests.test_studio_v3_connectivity -v
python -m studio.free_studio
python -m studio.production_queue --database /tmp/two_worlds_studio/jobs.sqlite3 --status
```

For an already Work-approved compiled **prompt pack** (schema 1, `scenes`
with `scene`, `prompt` and deterministic `seed`), prepare a plan **offline**:

```bash
python -m studio.free_studio --prompt-pack /private/my-prompts.json
python -m studio.cloudflare_flux --prompt-pack /private/my-prompts.json --scene S01
```

Neither command spends AI quota nor writes an image.

To store reviewed routing *plans* in a private SQLite queue, redirect the
offline output to a **private local path** then enqueue it:

```bash
python -m studio.free_studio --prompt-pack /private/my-prompts.json \
  > /private/my-routing-plan.json
python -m studio.production_queue --database /private/studio/jobs.sqlite3 \
  --route-plan /private/my-routing-plan.json
```

A SQLite file under /tmp is **not durable between separate GitHub Actions
runs**. For persistent autonomy, mount a private persistent volume or
implement a D1-backed queue adapter after D1 is actually provisioned.
The current queue manages state only; it does NOT automatically claim or
execute HTTP requests, models or publication.

To spend exactly **one** Cloudflare image request **only after checking account
billing/free quota, storyboard approval, license and privacy**, on a trusted
computer or private CI runner:

```bash
export CLOUDFLARE_AI_TOKEN='<read from secret manager>'
export CLOUDFLARE_ACCOUNT_ID='<read from non-secret GitHub variable>'
python -m studio.cloudflare_flux --prompt-pack /private/my-prompts.json \
  --scene S01 --reference /private/approved-character-portrait.png \
  --output-dir /tmp/two_worlds_private_art --execute-one
```

Never paste the token in GitHub source, logs or chat. The reference must be
a PNG/JPEG/WebP strictly smaller than 512x512 pixels. This is the **Cloudflare
model API input limit**, *not* the final generated image size.

The output is staged locally as `S01.png` plus `S01.art_review.json`;
it is NOT marked approved. Existing files reject a new paid/credit-consuming
request before calling the provider. The tool refuses to put unpublished
art anywhere under this public GitHub repository.

Cloudflare FLUX.2 REST API documented at:
https://developers.cloudflare.com/changelog/post/2026-01-15-flux-2-klein-4b-workers-ai/

## Production security

- Do not send unpublished canonical files or character portraits to AI Horde:
  requests are processed on arbitrary volunteer machines. The dispatcher permits
  Horde only for explicitly flagged **public generic, reference-free** jobs.
- Gemini free endpoints may have different data-use terms from paid accounts.
  OpenRouter `openrouter/free` is NOT a guarantee of zero retention or stable
  capacity; no auto fallback to paid models is implemented.
- Cloudflare model API may still incur usage charges depending on the owner's
  account plan and quotas. **Our code cannot enforce a universal hard $0 cap on
  an external billing account.** No autonomous inference runs by default.
- No model image similarity or character identity certification is claimed by
  a seed, a prompt, a token, or a returned PNG. Existing visual review remains.
- All production asset storage must be private; public GitHub Actions artifacts
  are deliberately not used for unpublished content.
- No auto TikTok/Metricool posting, no scraping websites to bypass API limits.

## Next integrations (NOT installed or proven)

The router has an honest inventory for ACE-Step music, Depth Anything V2
Small depth, SAM2 segmentation, Blender 2.5D camera, Whisper.cpp/Piper CPU
speech, Wan lightweight video, and Cloudflare Queues/D1/R2. These need
actual code, pinned model weights, rights checks, CPU/GPU benchmarks and
real media validation before production. Installing heavy models in public
GitHub Actions or treating an HTTP200 token as working inference is unsafe.

Acceptance remains **three original ARCHIVE 17 pilot MP4s** with 24 real
staged images, consistent characters, human-reviewed rights, Russian voices,
subtitles, covers and independent full decode QA. The new adapter alone
does **not** meet that acceptance criterion.
