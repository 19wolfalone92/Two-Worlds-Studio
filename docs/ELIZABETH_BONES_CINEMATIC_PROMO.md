# ELIZABETH BONES — cinematic promo upgrade (no paid APIs)

This is a **review-gated module of the existing Two Worlds Studio**, not a new repo or a replacement for its production pipeline. It does not change canonical TWO WORLDS scripts, story text, published accounts, or scheduled posts.

## Two worlds / brand

- **«Элизабет и Костя»**: carefully scripted gothic humor. Kostya is clearly fictional. Do not turn real families' grief or actual deceased people into jokes.
- **«Последний штрих»**: discreet, practical professional explanations; closeups of clean tools, white roses, clothes, cosmetic brushes. No real deceased bodies in promotional media.
- Elizabeth's portraits are **private reference assets**; do not upload personal portrait files to this public GitHub repository.

## Target: real 28–35 second Reel, 9:16

| Time | Required shot/action | Sound |
|---|---|---|
| 0–3s | Live-looking close-up, Elizabeth turns to camera in her studio | Short hook |
| 3–8s | Kostya enters, articulated head and jaw | «А у тебя бывают недовольные клиенты?» |
| 8–12s | Elizabeth actually reacts and speaks (not static photo with zoom) | «Костя, ты опять начинаешь?» |
| 12–15s | Kostya animated punchline | «Просто собираю отзывы!» |
| 15–17s | Rose petal transition, music switches mood | Instrumental bridge |
| 17–27s | Real moving hands preparing makeup brushes/flowers/clothing | Gentle, professional Russian narration |
| 27–33s | Portrait/brand end card | ELIZABETH BONES / «Чёрный юмор. Светлая душа» / Direct |

## What blocks the release

1. Movie contains <65% **real footage or character animation**. Photo zooms, Ken Burns, parallax-only, moving particles or kinetic captions cannot be passed off as actor movement.
2. No recorded/synthesized **natural Russian voice files**, no actual Kostya/Elizabeth dialogue, or unreviewed pronunciation.
3. Media provenance, usage rights or consent unresolved. Never commit private portraits or personal family materials to this public repository.
4. Human reviews *identity consistency, hands/lips movement, acting, voice listening, captions, taste and technical defects* before any release.
5. **No automatic paid model API calls, Kaggle GPU starts, merging to main, or publishing.** A future GPU-generated clip must have independently reviewed outputs and a valid license.

## Current local benchmark — prior 32-second Elizabeth Bones mockup

The previously assembled video has a real 720×1280 video stream and a music audio track, but **no natural Russian dialogue** and mostly still images with camera zooms. A sample frame-difference scan flags approximately **88% low-amplitude visual-change intervals**, characteristic of the slideshow approach. This is *not* a finished cinematographic promo. A basic FFprobe pass is not evidence of artistic quality.

## Run (CPU, no account, no credits)

```bash
python -m studio.promo_quality_gate --video /private/Elizabeth_Bones_Promo_Two_Worlds.mp4 --out /private/quality.json
python -m studio.promo_quality_gate --manifest /private/EB_Promo/project.json --out /private/provenance.json
python -m unittest tests.test_promo_quality_gate -v
```

The manifest gate requires 3–25 `shots`, fields `world: comic|care`, `kind: motion|still`, `asset` relative to the **private** manifest, `seconds`, `contains_real_deceased: false`, `rights_review: human_verified`; each `motion` also requires `motion_provenance: camera_footage|rendered_character_animation`. Spoken shots require `dialogue_text`, `voice` WAV/MP3 etc., and `voice_review: human_listened`. Root metadata includes `schema:1`, `id`, `language:ru`, `platform:instagram_reels`, `final_card_text`, `call_to_action`, `music_review:human_listened`.

`TECHNICAL_PREFLIGHT_PASS` is **not publication approval**. `inspect_video` is a conservative warning detector and cannot distinguish a subtle human performance from static scenery or certify lipsync. User approval and actual final MP4 are separate deliverables.

**First quality milestone:** one 5-second moving Elizabeth/Kostya dialogue scene with clean face/voice sync and consistent identity. Produce that on the existing private free-GPU pipeline with real model weights before calling the full promo complete.
