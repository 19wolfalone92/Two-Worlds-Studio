# FreeVideo-inspired T4 experiment (isolated, unmerged)

**Status: research prototype, CPU policy tests only. This is NOT a trained or merged hybrid neural model.**

Goal: selectively test transferable engineering ideas from [FreeVideo](https://github.com/FlashML-org/FreeVideo) (adaptive memory-aware planning, cheap first-pass preview, and per-run telemetry) **without installing FreeVideo, changing Studio production code, downloading H3 weights, or using Kaggle quotas without an explicit command**.

## What is included

- **Single-device CUDA probe** (via installed PyTorch) and conservative `Wan2.2 TI2V-5B` 9:16 preview/final geometry presets. Available VRAM controls the preset; these are **heuristics**, not measured memory guarantees. 4×T4 across two Kaggle sessions are **four independent workers** — memory cannot simply be pooled.
- **Explicit two-stage generation** through the existing `python -m studio.comfy_neural` API. A **human-reviewed**, SHA-256-bound preview is required before *any* final generation. No automatic retry, model download, publishing, paid API call or remote ComfyUI access.
- **Private JSONL real-run metrics** and grouped median timing. No fabricated speedups, anatomy scores, identity scores or claims that a different-resolution preview improves final quality.
- **No FreeVideo/VDN/MiniMax H3 code, architectures, weights or licenses incorporated**. H3 is excluded from the T4 runtime; this is a Wan pipeline and experiment, not a neural model merge. BF16 is not used on T4.

## Reproducible offline checks

From the repo root (Python 3.11+, no GPU required):

```bash
python -m unittest discover -s experiments/freevideo_t4 -p 'test_*.py' -v
python experiments/freevideo_t4/adaptive_pipeline.py plan --simulate-t4
```

`--simulate-t4` is visibly simulated and **cannot** be passed to a GPU command. Without it, `plan` reads one actual CUDA device via `torch.cuda`; Kaggle may expose multiple independent devices *inside the same session*, which must be scheduled separately.

## Optional manually authorized real experiment

Prerequisites: official Wan2.2-5B model weights installed in private ComfyUI; a reachable **loopback-only** server (use SSH tunnel if necessary); valid rights/license reviewed; private reference PNG; PyTorch and Studio dependencies installed. The repo is public: **all images, clips, receipts, human review and stats must live outside Git**. No automatic setup is provided.

```bash
# Inspect a real GPU; this does NOT generate or download anything.
python experiments/freevideo_t4/adaptive_pipeline.py plan

# Real preview: opt-in only; may consume Kaggle quota.
python experiments/freevideo_t4/adaptive_pipeline.py preview \
  --reference /private/eb01/shot.png --output /private/eb01/preview.mp4 \
  --prompt 'A skeletal character smoothly turns its entire head and shoulders while jaw and bones remain rigid.' \
  --history /private/eb01/timing.jsonl --accept-model-license --execute
```

Watch the resulting MP4. If correct, manually create `/private/eb01/review.json` (NEVER let software approve itself):

```json
{
  "schema": 1,
  "approved": true,
  "reviewer": "owner",
  "preview_sha256": "<sha256 of preview.mp4>",
  "reference_sha256": "<sha256 of shot.png>",
  "checks": {"identity": true, "anatomy": true, "motion": true, "no_morphing": true}
}
```

Only after visual inspection and that separate human action:

```bash
python experiments/freevideo_t4/adaptive_pipeline.py final \
  --reference /private/eb01/shot.png --output /private/eb01/final.mp4 \
  --preview-video /private/eb01/preview.mp4 --review /private/eb01/review.json \
  --prompt 'A skeletal character smoothly turns its entire head and shoulders while jaw and bones remain rigid.' \
  --history /private/eb01/timing.jsonl --accept-model-license --execute

python experiments/freevideo_t4/adaptive_pipeline.py stats --history /private/eb01/timing.jsonl
```

Both generated takes remain **GENERATED_REVIEW_REQUIRED**, never become published or approved final films. The preview review only authorizes a *separate* final generation; final must be visually evaluated again using existing Studio master gates.

## Acceptance / future integration rules

1. Baseline and experiment must render **same reference, prompt, seed, resolution, frame count and steps** for any legitimate speed/quality A/B comparison. The different-resolution preview is a cost filter, not evidence of a faster model.
2. Confirm T4 end-to-end completion with official weights and actual measured VRAM, wall-time and ComfyUI receipt. **No GPU acceptance test has been run in this branch.**
3. Human-review Elizabeth's face/hair and Kostya's nondeforming skull/jaw across the entire clip. Check native motion (not pan/zoom on a still), hands, background warping and sound independently.
4. Determine if the *total* spent on preview + accepted finals is lower than the baseline's discarded full-cost attempts, while maintaining final visual quality.
5. Only after independently demonstrated improvements, request a **separate** code review and selective port; **never automatically merge this branch into `main`**.

License research (do not copy model weights): [FreeVideo](https://github.com/FlashML-org/FreeVideo), [VDN paper](https://arxiv.org/abs/2609.20744), [official Wan2.2 ComfyUI](https://docs.comfy.org/tutorials/video/wan/wan2_2), and [the Studio's existing bridge](../../studio/comfy_neural.py).
