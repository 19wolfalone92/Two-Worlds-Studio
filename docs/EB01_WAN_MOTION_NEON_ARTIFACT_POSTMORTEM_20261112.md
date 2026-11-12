# EB01 — postmortem: Wan2.2 49-frame real motion, neon streak artifact

On 2026-11-12, native ComfyUI/Wan2.2 TI2V-5B GPU proof from private Kaggle
`wolf19loner92/eb01-v6-real-kostya-30e60bb07d4e` produced a real 49-frame
448 × 800, 24 FPS, H.264 MP4 (not a synthetic storyboard or pan/zoom).
GitHub Actions run: https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/43159416691

**The old technical QA incorrectly accepted the clip.** It had a video
stream, decoded all 49 frames, showed first-to-last image change and matched
SHA256 `15349e17d6f38576ece54330f7cbc6dbc9f8ac7f3560e377afa371cbf3ff4668`.
Yet visual review found neon-yellow/blue contamination at the bottom edge in
frames **46–48** (zero-indexed).

Independent CPU-only forensic check of decoded RGB frames, using the bottom
14% of each frame and fraction of pixels where max channel > 150 and
(max channel - min channel) > 65, gives:

| Frame group | Bright chromatic fraction |
| --- | ---: |
| 0–44 | near 0.0001 |
| 46 | **0.0971** |
| 47 | **0.1008** |
| 48 | **0.0639** |

This is a sudden color corruption, NOT normal character acting.

## Root-level fix

1. `studio/eb01_motion_artifact_gate.py` now examines the **decoded frames**,
   rejects sudden interior color-streak artifacts, and detects terminal
   contamination without a GPU run. Accepted baseline is calculated from
   the first 70% of the sequence; the threshold is the maximum of 0.028,
   median + 0.025, and median × 12.
2. The *only* safe automatic salvage option is trimming contiguous corrupted
   **terminal** frames, producing a new separately hashed review MP4. No
   morphing, re-generation, arbitrary splice, or full-episode approval.
3. `studio/v6_production.py` refuses any neural motion take flagged by the
   temporal gate *before* accepting its provenance receipt in the native
   compositor. FFprobe passing alone is no longer a creative-quality signal.
4. CPU-only tests recreate the exact 46+3 pattern, a mid-clip failure and a
   21-frame H264 corruption case with safe suffix trim. No GPU/quota consumed.

The 46-frame salvage SHA256 is
`5b77cba4f49b70d84affa71e5d2238235b9997923c69e161737bf00abb7fa03a`.
It is a **technical preview only**, **not a full finished scene**: jaw-sync,
body acting, continuity across all seven shots and six Russian dialogues
remain unapproved.

No social posting; no proprietary weights or source portraits are committed.
