# STORY GLITCH / TWO WORLDS — free autonomous Studio v2

**State (2026-11-06): production engine modules and offline tests, NOT finished episodes.**

This upgrade extends the existing source tree, not a separate service, and
keeps Williams Bot / Williams AI completely independent.

## Fixed production contract

- ChatGPT Work alone authors and locks screenplay. **Studio never writes,
  changes, summarizes or secretly replaces a script**; the accepted ARCHIVE 17
  master is version 1.1.
- ARCHIVE 17: 30 episodes, first pilot “ДЕВОЧКА НА ФОТОГРАФИИ” is
  **3 parts / 24 original shots** (8 per part), publish sequentially only
  after complete production, rights review and actual approval.
- Film: **1080x1920 / 30 fps / H.264 + AAC / UTF-8 Russian SRT + baked Russian subtitles**;
  three separate MP4s and **distinct color covers**, plus an immutable
  receipt and one review ZIP.
- Camera: restrained cinematic Ken Burns with subpixel interpolation,
  easing, no integer step jitter; authentic motion clips only when supplied.
  The camera cannot magically animate still photographs or real facial acting.
- Audio: recorded or legitimately generated **Russian speech** with stable
  fictional voice cast; measure actual WAV length and reject overlong voice
  tracks rather than silently truncating or inventing timing.
- No paid service, TikTok/Metricool writes, unapproved human impersonation,
  secrets in a public repo, fake GPU media or automatic self-approval.

## Connected modules

| Stage | Actual component | Contract |
| --- | --- | --- |
| Locked Work story | `studio.work_contract` | No unauthorized text changes |
| Visual prompts | `studio.art_director` | Reproducible scene seeds + reference metadata |
| GPU image generator | `studio.comfy_kaggle`, `studio.art_director` | Private Kaggle T4, local ComfyUI API, model/PNG receipts |
| Hugging Face | `studio.hf_hub` | Read-only token + license metadata check (no paid inference) |
| Private GPU output | `studio.gpu_asset_handoff` | Verify nonce, ComfyUI job ID, hash, PNG, image size, stage **unreviewed** |
| Russian TTS (optional) | `studio.russian_qwen_tts`, `studio.narration_batch` | Qwen3-TTS, no script editing or cloning, WAV SHA/timing, private GPU required |
| Measured subtitles (optional) | `studio.subtitle_align`, `studio.hybrid_master` | Real CTC-spoken word timestamps with a staged model, or editor-entered `subtitle_cues`; never guessed timings |
| Performance quality gate | `studio.animation_quality` | Reviewer-signoff and actual character motion/grounding data; warns/rejects sticker overlays |
| Kaggle model cache | `studio.model_cache` | Bounded verified transient copy of mounted weights; no persistent duplicate |

| English TTS | `studio.kokoro_tts` | Optional Kokoro for approved English text |
| Montage | `studio.hybrid_master`, `studio.smooth_camera` | Frame-accurate, FFmpeg + Pillow, optional large burnt-in ASS captions, safe-zone placement, speech-triggered music ducking, SRT |
| Three-part batch | `studio.triptych_batch` | 3 × 8 shots, separate covers, ordered 1/3–3/3, atomic all-or-nothing ZIP |
| Release gate | `studio.release_gate` | Real H.264/AAC 1080x1920/30, SRT timing, first **three seconds** and cover, optional full decode |
| Publication | **Not automatic** | Only approved, reviewed MP4s; existing schedules unchanged |

## Ingest REAL ComfyUI image

After an authenticated private Kaggle output download, keep the ZIP and receipt
together on a **private** filesystem, then:

```bash
python -m studio.gpu_asset_handoff \
  --archive /private/TWO_WORLDS_ARTIFACTS.zip \
  --receipt /private/two_worlds_job_receipt.json \
  --nonce comfy-<the-exact-GitHub-run-id>-1 \
  --dest /private/inputs/scene_S01
```

The importer does not approve the source or claim stable faces. The default
SDXL API graph is a **single-scene technical test**, NOT an IP-Adapter or
multi-character reference-locked film pipeline. The real pilot needs approved
visual bible references before GPU batch generation.

## Three-part production input

`triptych_batch` consumes an existing script-derived **shot manifest** plus
real artwork and audio. These files belong in a local private workspace,
**not** in this public GitHub repository.

Example `private/ARCHIVE17/pilot.json`:

```json
{
  "schema": 1,
  "type": "three_part_story",
  "job_id": "ARCHIVE17_GIRL_IN_PHOTO",
  "story_title": "ДЕВОЧКА НА ФОТОГРАФИИ",
  "language": "ru",
  "editorial_approved": false,
  "rights_approved": false,
  "final_human_review_approved": false,
  "parts": [
    {"part": 1, "manifest": "P01/shot_manifest.json", "cover": "P01/cover.png", "expected_shots": 8},
    {"part": 2, "manifest": "P02/shot_manifest.json", "cover": "P02/cover.png", "expected_shots": 8},
    {"part": 3, "manifest": "P03/shot_manifest.json", "cover": "P03/cover.png", "expected_shots": 8}
  ]
}
```

Each `shot_manifest.json` uses the **existing**
`docs/HYBRID_VIDEO_STUDIO.md` schema, with width 1080, height 1920, fps 30,
8 image/clip shots, real Russian voice WAV tracks, \`burn_subtitles: true\` for all parts, captions, individual source
checksums (`media_sha256` and `voice_sha256`), optional
`requires_voice: true`, optional truthful phrase-level `subtitle_cues`
with local `start/end/text` offsets measured from real audio,
subtle camera moves, and no placeholders.
`--master` additionally requires explicit per-shot/editorial/rights/voice
approvals, a private `work_script_file` path and its real matching
`work_script_sha256`. No code can fabricate approval. Cache fingerprints
**every** input source image, WAV, cover and manifest: a changed byte invalidates
the cached film, even when the JSON manifest itself was not edited.

CPU preflight (no render):

```bash
python -m studio.triptych_batch \
  --package-manifest private/ARCHIVE17/pilot.json \
  --output-dir private/ARCHIVE17/reviews --preflight
```

Atomic 3-film review export (only when **all real assets exist**):

```bash
python -m studio.triptych_batch \
  --package-manifest private/ARCHIVE17/pilot.json \
  --output-dir private/ARCHIVE17/reviews --full-decode
```

Expected files in `ARCHIVE17_GIRL_IN_PHOTO_review/`:
`PART_1_of_3.mp4` through `PART_3_of_3.mp4`, matching SRT files, three
covers, a technical `delivery.json`, individual FFmpeg QA receipts and a
review ZIP. A second identical command returns `cached_verified` after
validating all three existing files again; changed source fails closed.
Never writes to TikTok or changes previously scheduled posts.

## Verification vs. claims

- A passing GitHub CI proves **offline code tests**, not image beauty,
  convincing face identities or successful TTS inference.
- The Kaggle ComfyUI backend **did run** on GPU previously; PNG readback
  via `kaggle kernels output` encountered HTTP **429**. The new worker stores
  weights in transient `/kaggle/temp` and readback limits transfers to ZIP
  plus receipt. Actual private output still requires verification.
- IP-Adapter/ControlNet for recurring faces, photorealistic actor motion,
  24 approved source images, 3 production-ready Russian WAV tracks and final
  three MP4s **are not yet proven or present in main**.
- Qwen3-TTS and Hugging Face Hub preflight were separately code-tested;
  model access and Russian human voice quality must be confirmed live.
- Keep **public project** free of unreleased screenplays, source references,
  voices, and unfinished images. Store private asset ZIPs on Kaggle or a
  private approved storage location; do not upload them to public CI artifacts.
- Existing TikTok and Metricool schedules are unchanged.

## Continuity ledger and immutable review-cache hardening

The pilot's three 8-shot manifests may contain a per-shot `continuity` object
with review-controlled reference identifiers. The offline
`studio.continuity_ledger` verifies **declared** continuity:

```json
{
  "continuity": {
    "location": {"id": "nina_kitchen", "layout_sha256": "<64-hex>"},
    "cast": [{
      "body_id": "Nina", "age_variant": "adult",
      "reference_sha256": "<64-hex>", "wardrobe_id": "winter_coat"
    }],
    "props": [{"id": "photo_17", "reference_sha256": "<64-hex>"}],
    "transitions": []
  }
}
```

Across the complete 24-shot sequence, the same location ID must keep the
same layout SHA; the same character and age variant must keep the same
reference SHA; recurring props must retain reference SHAs. Wardrobe/age
changes require explicit `transitions` containing a meaningful reason and,
for a master, an identified reviewer. Missing metadata is visible in preview
reports but is prohibited in `--master`. This is a **metadata audit only**;
it does not visually recognize the faces, correct reference images, lighting,
camera framing, real clothing, or actual dramatic performance.

The three-part builder now enforces **exactly eight real shots per part**.
Cache reuse fingerprints source manifests, covers, audio, **music** and all
source stills/clips. It also checks the **actual exported MP4, SRT, cover,
and ZIP hashes** before reporting `cached_verified`. Any changed source
means an existing review directory is rejected rather than silently reused.
To export a new revision, select a new job_id/output folder (do not overwrite
the first deliverable). The bundle SHA is a detached receipt field in the
on-disk `delivery.json`, avoiding circular ZIP self-hashes.

## Tests

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Human creative QA remains mandatory: face identity, story prop continuity,
natural Russian acting, emotional progression, frame jitter and hooks,
first 3 seconds, background integration, licenses, cover typography.

## Added cinematic audio and mobile captions

The three-part pilot requires `burn_subtitles: true` in every source shot manifest.
FFmpeg burns a legible Cyrillic-capable ASS overlay into the H.264 frames,
positioned above the bottom platform UI (approximately 16.5% screen height),
with pale text and a contrasting outline. An independent UTF-8 SRT is always
written for review/reuse. When editor/CTC-measured `subtitle_cues` exist,
these exact timestamps are used; otherwise, whole-shot captions are used
without pretending word-level alignment.

If an approved music track and narration WAVs are both supplied, FFmpeg now
applies **sidechain compression** to lower the soundtrack during actual speech,
followed by peak limiting. Neither background music nor source voice WAVs are
changed or fetched. This is a technical approximation, not a substitute for
listening to the final Russian mix on a phone. The ordinary hybrid editor
keeps `burn_subtitles: false` by default for backwards compatibility.
