# TWO WORLDS STUDIO — deep neural-video audit, 2026-11-11

**Verified source:** public repo `19wolfalone92/Two-Worlds-Studio`, main at
`555c79724e1b794e08636281cc888ea10bfcb809`, plus draft PR #60
(`studio-5/verified-cinematic-motion` at `eaaae4f61758bbe7ede0c8fd7f5c6da000cdf0d6`).
This document records a source audit and researched upstream configurations,
**not a successful real GPU render**. No weights, paid API requests, videos,
characters or private scenarios are committed to the public repository.

## Executive findings

| Severity | Verified code / symptom | Consequence | Remedy |
| --- | --- | --- | --- |
| **CRITICAL** | `studio/neural_motion.py` stages the legacy **LTX-Video 2B 0.9.6 distilled**, not the latest LTX-2.5; `studio/neural_character.py` in PR #60 only imports EchoMimicV3 Flash or older Wan2.2 Animate-14B. | An apparent high-end AI studio can still deliver the old low-detail motion backend. | Use verified modern model-specific generators, not a global sharpening switch. Model inventory and CLI plan added in `studio.video_ai_plan`. |
| **CRITICAL** | No evidence of actual GPU inference with the newer character adapters in draft PR #60. Its own report explicitly marks motion/cheek/material review as unresolved. | Offline CI green does not prove natural movement or photorealistic realism. | Owner-reviewed video-to-video benchmark on private assets, then 3-seed A/B and lock a passing worker model revision. |
| **HIGH** | `studio/deformable_actor.py` uses 2D weighted displacement over a single RGBA image; `studio/character_engine.py` builds sprite rigs, not volumetric biomechanics. | Tilts/bends produce disconnected cloth or half-body warping, impossible occlusions, unnatural object contact. | Full-body driving video + diffusion-driven character animation; use rigs only for intentionally stylized motion comics. |
| **HIGH** | `studio/facial_actor.py`, `studio/lipsync.py` and PR #60 patch-based performance can animate lip/facial regions, but do not inherently synchronize head, neck, shoulders, arms, cheek volume and jaw anatomy. | Speech looks like independently moving lips over a frozen face. | Drive facial expression/head pose/neck/body together; lip-sync as finishing correction, not the primary acting engine. |
| **HIGH** | `studio/production.py` still defaults to **540×960/12 FPS**, while `studio/hybrid_master.py` uses 24–30 FPS and PR #60 declares a mobile profile. | Calling the wrong entry point yields visually old, choppy clips despite 5.0 features. | Treat `python -m studio.cli produce` as reviewed hybrid pathway. Reject old renderer for premium acting shots; document preset explicitly. |
| **HIGH** | Kaggle T4 devices have ~16GB memory **per GPU**, not an automatic pooled 32GB. Official LTX-2.5 DFR requires much more headroom, Wan Animate-2 reports multi-A800 tests. | Simply installing weights on T4 cannot guarantee 2026 flagship quality. | Start with **Wan2.2 TI2V-5B on ComfyUI native offload**; request higher-memory GPU only for demonstrably better generations. |
| **MEDIUM** | `studio/motion_qc.py` calls the backward optical-flow stage even when the forward stage loses all tracking points, and consumes RANSAC inliers without checking None. | Blur/occlusion can crash verification rather than return a review flag. | Fail closed with empty tracks and preserve review status. Fixed with offline regression tests in this audit branch. |
| **HIGH** | Most quality checks use motion texture/proxy metrics, not an independently verified actor-physics or identity score. | A passing numerical threshold cannot prove fingers, mouth, garments, facial expression or physics. | Human evaluation on full decoded MP4 alongside reference and driver, and explicit physical-contact/performance QA. |
| **MEDIUM** | The project has many production entry points, model assumptions and legacy branch code, plus 24+ files altered in draft PR #60. | Configuration divergence, inconsistent outputs and stale cache variants. | Pin every clip by checkpoint revision, prompt, seed, source hashes, FPS, output and review provenance. Converge on the hybrid master. |

## Verified upstream AI options and official inference defaults

The word *open* must distinguish Apache/MIT code from gated or community
model weights. Never copy a closed generator or remove a license restriction.

| Family | Actual strengths | Official relevant settings | Practical use |
| --- | --- | --- | --- |
| **Wan2.2 TI2V-5B (Apache-2.0)** | 720p text/image to video, movement and cinematic composition. | `ti2v-5B`, `--size 1280*720` (pixel area; I2V follows input aspect), **121 frames, 24 FPS, 50 sampling steps, flow shift 5.0, CFG 5.0**. | Primary test for 9:16 authored artwork. Official runner ~24GB 720p, but **ComfyUI documents native 8GB offloading**; use its current template for Kaggle T4. |
| **Wan2.2 I2V-A14B (Apache-2.0)** | More complex high-fidelity I2V, significantly heavier. | `i2v-A14B`, 720p, **81 frames at 16 FPS, 40 steps, shift 5.0, MoE expert guidance (3.5, 3.5)**. Do not flatten guidance into one scalar. | Candidate for higher-memory GPU and review; 16 native FPS is not 24 native FPS. |
| **Wan-Animate-2 (Apache-2.0)** | End-to-end character animation **from a driving video**, reduced need for intermediate pose extraction, improved identity stability. | Base **40 steps**; distilled **10 steps, CFG 1.0, Euler**. | Best research direction for anatomical movement. Official repo documents tests on **2× A800 at 480p, 8× A800 at 720p**, not free T4 proof. |
| **LTX-2.5 DFR (Community model license)** | Native audio-video, detailed multi-shot, high-fidelity diffusion pass + spatial detailing IC-LoRA. | Official DFR pipeline uses **121 frames, default 1024×1536, 24 FPS**, model-managed stages and trained detailing LoRA. No arbitrary 40-step retrofit. | Studio quality contender if suitable GPU/storage is provided. Official system minimum **32GB VRAM / 32GB RAM / 100GB storage** for open-source setup. Model files ~66 GiB in upstream quick start; license acceptance required. |
| **MuseTalk 1.5 (MIT code, separate model/dependency licenses)** | High-quality speech retiming around a **256×256 face crop**. | Use aligned audio + moving original video, then inspect jaw, cheeks and masks frame-by-frame. | **Finishing tool only**, not full-person animation. |
| **LivePortrait** | Pose/gaze/expression transfer for portrait-centric scenes. | Driving performance, stitching/retargeting, safe eye/lip editing. | Complements, does not replace full-body Wan Animate. |
| **HunyuanVideo 1.5 (Tencent community)** | Powerful text/image-to-video with low-memory offloading. | 720p I2V **50 steps, CFG 6, flow shift 7**; official 14GB memory floor with offload. | **Hold for legal review:** published license restricts some regions, including the EU. |

Official upstream sources (review before installing):
- Wan2.2 source / sampling: https://github.com/Wan-Video/Wan2.2 and `wan/configs/wan_ti2v_5B.py`, `wan/configs/wan_i2v_A14B.py`.
- ComfyUI native low-VRAM workflow: https://docs.comfy.org/tutorials/video/wan/wan2_2
- Wan-Animate-2: https://github.com/Wan-Video/Wan-Animate-2
- LTX-2.5 & DFR: https://github.com/Lightricks/LTX-2
- Official hardware requirements: https://docs.ltx.io/open-source-model/getting-started/system-requirements
- MuseTalk: https://github.com/TMElyralab/MuseTalk
- LivePortrait: https://github.com/KlingAIResearch/LivePortrait
- HunyuanVideo 1.5: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5

## High-fidelity production policy (not a mere resolution increase)

1. **Input:** lock identity, hairstyle, clothes and scene lighting with one reference package for every character and location. Supply full height 9:16 photographic frames, not transparent stickers.
2. **Performance:** capture/choose a short real driving performance that contains the intended **whole-body action**. If a character leans, the face, shoulders, pelvis and feet must move consistently with its centre of mass. Generate per shot (roughly 2–5 s), not one 60-second clip.
3. **Native rendering:** keep each model's trained frame schedule, guidance and FPS; vary seed to pick the better take instead of blindly raising CFG, steps, resolution or interpolation. Never call a static-frame clone neural motion.
4. **Speech:** decide whether the selected video model generates speech together with the image (LTX) or the studio supplies its already locked Russian Qwen WAV (Wan + lip/facial engine). Do not combine both independent voices into one scene. Retiming must be based on the actual WAV duration and phonemes.
5. **Lighting and physical realism:** prefer a full character generation over 2D deformation when turning or speaking. Match shadows, contact, parallax and exposure with the scene; test hand-object reach and cloth physics. A static camera does not excuse a static face/body.
6. **Temporal QC:** inspect all decoded frames for image-to-image identity drift, flicker, duplicate/frozen frames, cut boundaries, facial shape, lip/audio sync and hands. Use mouth, jaw, cheek, eyes, torso, hand and background independently anchored regions. Any missing evidence is **REVIEW REQUIRED**, never PASS.
7. **Mobile export:** render preferably at **native 24 FPS** in the generation model and deliver 576×1024 or 720×1280 when quality permits; re-encode H.264 CRF 17–20, `yuv420p`, `+faststart`. Never equate an upscale from 480p or duplicated frames with genuine model detail/motion. 1080×1920 is an optional clean final scaling step, not a fidelity cure.

## New offline quality planner

```bash
python -m studio.video_ai_plan --list

# After separately staging OFFICIAL Wan2.2 code and legal Wan2.2-TI2V-5B weights:
python -m studio.video_ai_plan \
  --profile wan22_ti2v5b_quality \
  --runtime /private/models/Wan2.2 \
  --weights /private/models/Wan2.2-TI2V-5B \
  --reference /private/scenes/shot_01.png \
  --output /private/takes/shot_01.mp4 \
  --prompt "The woman turns her whole body gently to face the visitor, shoulders and neck follow the motion; realistic skin and coherent lighting, locked camera." \
  --seed 42 --free-vram-gib 24
```

The planner prints a structured `command_argv` (never shells it), the exact
native frame clock, the hardware caveat and `inference_performed: false`.
It blocks unsafe aspect ratios, transparent sprites, accidental overwrites,
unsupported model runners and unsupported VRAM claims. Execute only on a
private, adequately resourced worker. For Kaggle 16GB T4 use the
**current ComfyUI 5B template with offloading** instead of pretending that
the unmodified 24GB official CLI is guaranteed to run.

## Release acceptance benchmark

Select **three reproducible 5-second private shots** with exact shared
reference frames: (A) a woman leans forward, reaches, lifts an object,
then stands; (B) a man says a Russian line with jaw/cheeks/eyes/shoulder
expression; (C) both people exchange eye contact and react naturally.
For each model/seed, save native MP4, prompt, checkpoint revision, driver,
audio WAV, full frame count, VRAM/log output and final source hashes.
Record defects per second: *partial-body warp; lip-only puppet;
floating feet/hand; frozen cheeks/neck; flicker; identity drift;
unnatural lighting; audible lip offset*. No model gets `master` unless
both automated QC and owner visual/audio review pass.

**Still unverified:** downloading/staging 2026 model weights, real
Kaggle/ComfyUI availability, GPU VRAM peaks, and improvement on the user's
actual video reference. This audit is an executable preparation and code
safety fix, not proof that a successful flagship model is already installed.
