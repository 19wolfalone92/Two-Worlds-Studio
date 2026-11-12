# TWO WORLDS STUDIO V6 — Quality First (12.11.2026)

**Status:** source-code refactor + CPU checks; NO newly produced neural video and NO
artwork or weights uploaded to public GitHub. Old 23-second Elizabeth Bones E01
video is visually REJECTED and cannot be distributed as approved production.

## Verified root failure in the previous E01
1. Seven independent SDXL prompt generations with no common image-conditioned
   character reference. Kostya changes from skeleton to human; Elizabeth changes
   face/hair/wardrobe.
2. Camera pan/zoom of static images was reported as character animation.
3. MP4 codec/SRT/6-audio-count assertions tested packaging, not visual content.
4. FP16 Qwen3-TTS on Kaggle Tesla T4 triggered a CUDA device-side assertion;
   precision selection must use native BF16 where supported, otherwise FP32.

## The new one-route production policy
- **Art:** private canonical master for Elizabeth and for *anatomically skeletal*
  Kostya; reference-conditioned image generation, not independent text sampling.
  Existing `studio.character_factory` and `studio.comfy_reference_guard` are
  retained. A ComfyUI graph with an unconnected LoadImage is rejected; the
  IP-Adapter must influence the actual sampler. Neither graph validity nor a
  SHA match proves the pictured person is correct: human image review is
  mandatory, per shot, before any GPU animation.
- **Motion:** primary `wan22-5b` uses existing REAL `studio.comfy_neural`,
  the official ComfyUI Wan 2.2 TI2V-5B model template and official model
  weights held privately, never copied into GitHub.
- **Pilot only:** one original source still -> one 49-frame, 24 fps, I2V video.
  There is no automatic multi-seed fanout, no retry, no long batch, no paid API.
  The master preview is blocked until that short clip is visually accepted.
- **Full:** seven real, native 24 FPS I2V clips, each matched against their
  original frame SHA and a genuine ComfyUI inference receipt. AI output
  filenames, hashes, model identity and clip duration must all agree.
- **Voice:** six exact Russian Qwen3 source recordings; Qwen precision patched
  so T4 does NOT use unsupported BF16 or FP16 inference.
- **Composition:** the already established `studio.hybrid_master.produce`
  handles approved video clips, not static-frame camera movement, and creates
  MP4+SRT+technical report. V6 forbids any static fallback.
- **QA:** visual reviews are explicit, bound to SHA, separate for stills and
  motion. A review may be owner- or appointed-human authored; **never
  automatically invented** by a pipeline. Even a completed technically valid
  video stays `RENDERED_REVIEW_ONLY`, not "published" or "final approved".

## Backend audit and selection
| Backend | Decision | Reason |
| --- | --- | --- |
| Wan2.2 TI2V-5B / ComfyUI | **PRIMARY** | Official ComfyUI low-VRAM I2V template, supports offload; integrated already as actual local execution in `studio.comfy_neural`. |
| Wan-Animate-2 | **OPTIONAL HEAVY** | Whole-body driven acting; real GPUs and driver video required. Existing `studio.wan_animate2_run` retained. |
| MuseTalk 1.5 | **OPTIONAL FACIAL FINISH** | Good for a *human* face after actual whole-body animation, but NOT proven to animate a bone skull or cloth/body physics. Do not enable blindly for Kostya. |
| InstantID | **NOT PROMOTED AS DEFAULT** | Human face consistency, not a bone skeleton identity model. Noncommercial/research restrictions on published face weights; review licensing first. |
| CogVideoX / Mochi | **RESEARCH ONLY** | Additional operators/weights/licensing and GPU validation required; no zero-evidence installed claims. |
| LTX-2.5 DFR | **HIGH-VRAM OPTIONAL** | Official open-source guide says 32 GB+ VRAM and 100 GB+ disk for ComfyUI. Existing integration kept separately, not selected on T4. |
| HunyuanVideo full original | **NOT DEFAULT** | Official original repo requires ~45–60 GB for cited resolutions; different Hunyuan 1.5 is a separate license/hardware review. |

Official references:
- https://docs.comfy.org/tutorials/video/wan/wan2_2
- https://github.com/Wan-Video/Wan2.2
- https://github.com/TMElyralab/MuseTalk
- https://github.com/instantX-research/InstantID
- https://github.com/zai-org/CogVideo
- https://github.com/genmoai/mochi
- https://docs.ltx.io/open-source-model/integration-tools/comfy-ui/
- https://github.com/Tencent-Hunyuan/HunyuanVideo

Copying proprietary commercial models, misrepresenting model installs, or
vendoring multi-GB weights to the public repo would make the Studio less
reliable and may violate rights. V6 reuses already implemented working
adapters and adds strict, fail-closed orchestration rather than duplicating them.

## Removed obsolete technical-only routes
- `studio/rig_demo.py` — a purely synthetic animation demo, not genuine AI.
- `.github/workflows/wan-real-gpu-proof.yml` and
  `.github/workflows/wan-real-gpu-readback.yml` plus their one-shot
  `ops/wan_*.trigger` files — old public engineering-fixture proof flows.
- The downloadable synthetic MP4 proof produced by the default CI workflow;
  ordinary unit/renderer testing remains available. CI cannot claim a film is
  ready.

**Preserved, not deleted without dependency analysis:**
`studio.hybrid_master`, `studio.comfy_neural`, `studio.model_install`,
`studio.russian_qwen_tts`, `studio.comfy_reference_guard`,
`studio.character_factory`, subtitle/audio mastering, source integrity and
existing story-production workflows. Legacy sprite rigs and old LTX are no
longer eligible as V6 premium final-video engines; some remain in source so
existing educational motion comic tests and unrelated projects do not break.

## How to use (no background promises)
1. Copy `projects/ELIZABETH_BONES_E01_V6_PRIVATE_TEMPLATE.json` into a folder
   **outside the public Git repository**. Produce 2 approved PNG canonical
   images and the 7 reference-conditioned PNG scenes locally/privately.
   Replace each placeholder with the actual SHA-256 of the image/audio.
2. Stage six original Russian WAV recordings under the storyboard's voice
   paths, adjust scene lengths to cover the complete spoken WAVs.
3. Create `visual_review.json` beside the private assets:
   `schema=6`, `episode=ELIZABETH_BONES_E01`,
   `storyboard_sha256=SHA256(storyboard.json)`,
   `human_visual_review=true`, `reviewer=owner`, `approved=true`,
   `reference_approved={elizabeth:true,kostya:true}`,
   `visual_checks={S00..S06:{identity:true,wardrobe:true,anatomy:true,
   lighting:true,composition:true,kostya_is_skeleton:true}}`.
   Only fill those fields **after personally seeing the images**, not in CI.
4. Offline preflight: `python -m studio.v6_production plan --storyboard /private/storyboard.json`.
5. ONE actual optional GPU pilot on a real installed local ComfyUI server:
   `python -m studio.v6_production pilot --storyboard /private/storyboard.json --review /private/visual_review.json --assets /private/eb01 --shot S02 --execute-local-gpu --accept-model-license`.
   Omitting `--execute-local-gpu` consumes no quota.
6. Watch the real pilot. Keep the resulting MP4 and its
   `S02.receipt.json`. Add actual clip/receipt SHA to a separate private
   `takes.json` with `schema=6`, storyboard SHA and `shots` mapping.
7. Author `motion_checks` for **each** shot with its MP4 SHA, reviewer,
   approved flag and all five true checks. Set `pilot_approved=true` only
   after reviewing real character movement. Repeat I2V only for the other
   approved still frames as intentionally directed actions (never auto-run 7).
8. `python -m studio.v6_production render --storyboard ... --review ...
   --takes ... --assets /private/eb01 --name EB01_REVIEW.mp4`.
   This operation refuses missing or synthetic clips and never publishes.

## Current blockers
- No new, owner-approved canonical images for both actors in the repository,
  and no verified, identity-conditioned 7-shot sequence approved for this V6.
- No owner-reviewed REAL Wan I2V pilot with natural bone jaw/head/body motion.
- Existing E01 MP4 still fails visible identity and material continuity and
  cannot be fixed by FFmpeg or passing a technical test.

The purpose of these barriers is to **stop wasting GPU quota** rather than
run an expensive bad full video again.
