# Elizabeth Bones — deep root-cause audit and corrective actions (2026-11-14)

## Verified scope and truth boundary

Project: `19wolfalone92/Two-Worlds-Studio`, working branch `fix/eb01-s02-private-raw-preservation-20261113`.
`main` remains unchanged by this audit. Social publication forbidden.
The user's previously rejected 23.458333-second film is **NOT** an accepted release.
There is still **NO verified completed new high-quality 40+ second MP4**.
The source-gate human artistic approval requirement cannot be bypassed by CI.

### Evidence (actual, not simulated achievements)

- Full repository global regression (zero GPU): **579 tests pass, 4 skipped**, actual GitHub Actions **SUCCESS** https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/43600286616. Full `studio/`, `tools/`, and `tests/` Python compilation passed. Tests include full-frame FFmpeg and CPU contracts.
- Independent model patch reproducibility: exact official LTX-Video revision `50f0aef1297574a4b29acbdb874e3a363a6d5f2e` cloned, all production-actual AST patch anchors applied, resulting upstream Python **compiled**, 13 genuine native stage markers and actual **8-step** distilled schedule verified, 21 CPU tests pass: https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/43599934080.
- GitHub Actions original owner-review anti-fake safeguards: https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/43600076670.
- Independent Kaggle account secret authentication: `KAGGLE_API_TOKEN` and `KAGGLE_API_TOKEN2` both exist in GitHub Actions for THIS repo and pass individual read-only Kaggle API checks. They resolve to **two distinct Kaggle accounts**; 6 tests pass. **Private original studio source dataset access by second account is DENIED**: https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/43598931442. The second account **has not** performed GPU film generation, originals have **not** been copied, and no quota bypass has been attempted.
- Actual previous real LTX image-to-video 33-frame GPU attempts: GH run `43559988099` and `43579646805` failed. Both were real GPU jobs with bounded 1400-second native sampler timeouts after being scheduled; the second was initially QUEUED, then actually executed and hit the same timeout. Detailed read-only real Kaggle traceback: https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/43597728095.

### Root-cause corrections committed

1. The original studio's GPU preflight assumed 20/21 steps while the pinned LTX 2B distilled model actually uses **8 steps**. `studio/eb01_render_feasibility.py` and `.github/workflows/eb01-ltx-short-pilot33.yml` now check 8, reject misleading step budgets; Boolean `True` no longer passes numerical GPU-VRAM checks.
2. Pinned LTX 2B device staging: stage CPU model loading, GPU text encoding, GPU VAE encoding of ORIGINAL character reference, VAE offload, GPU transformer denoise, transformer offload, GPU VAE final decoder. `tools/kaggle_eb01_ltx2b_s02_fallback.py` strict, no synthesis shortcuts; auxiliary prompt enhancement Florence-2 + Llama-3B disabled, native 8-step schedule kept.
3. Missing LTX stage telemetry made previous 1400s failures impossible to localize. New `tools/eb01_bounded_inference.py` monitors authentic subprocess, strictly bounded process-group kill, real-time **stage-only** heartbeats, private logs, no raw portrait paths/audio prompt leak into public Actions logs. All 13 stages are verified against official model source in `tools/eb01_ltx_official_compatibility_audit.py`. **Actual live CUDA success still unproven.**
4. `studio/eb01_motion_artifact_gate.py`: global regression exposed a real bug where a conservative RGB spectral anomaly classifier accidentally vetoed *provably terminal-only* corrupt-frame trimming. Fixed to allow removal only when all flagged frames are in a known terminal discard suffix, then independently re-decode the cleaned clip; **any interior neon still fails closed**.
5. `studio/eb01_cinematic_delivery_gate.py`: reject missing or invalid real MP4 SHA256 and missing/Boolean/negative supposed `0 neon frames` evidence; original identity, unique seven performances, lip-sync and human owner signoff remain compulsory.
6. Private frames: both native GPU and watcher workflows now **forbid GitHub artifact media uploads when repo is public**. No original characters/six Qwen WAV files may leak into public artifacts; private Kaggle retains raw materials.
7. `tools/eb01_gpu_watchdog.py` distinguishes QUEUED (compute availability), RUNNING, ERROR (real model), COMPLETE (requires real frame SHA+FFmpeg), including instant root categories. `.github/workflows/eb01-ltx-short-pilot33.yml` now has an on-failure private-log diagnostic step for future runs that stores only redacted category and last true stage, not original data.

### Second authorized Kaggle API token

The protected `KAGGLE_API_TOKEN2` credential is **present and authenticates** separately. It does **not** yet have `datasets.get` visibility on `wolf19loner92/eb01-v7-private-production-sources`, so it cannot legally and technically be used for a source-locked Elizabeth/Kostya I2V job. **Do not publish the dataset or copy the 2 private portraits or 6 voices to work around access control.**

Kaggle officially supports **private dataset collaborators (Can view)** through dataset **Settings → Sharing**. The family member should operate their **own** account; Kaggle terms restrict account sharing and people operating multiple logins. See https://www.kaggle.com/docs/datasets and https://www.kaggle.com/terms. Re-check the independent dataset ACL before any future work under a collaborating account; no GPU from token2 was dispatched.

### Actual current production status

After the CPU fixes, exactly ONE new 33-native-frame **owner-account** LTX S02 private GPU quality pilot was initiated from the currently fixed commit through:
https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/43600628791

It is only a **GPU validation pilot**, not the film. The workflow must verify original SHA, actual 33 decoded frames, FPS24, model timing and every-frame RGB before any cinematic owner review. No lengthy 125-frame T4 job is authorized. A green CPU test or Kaggle `push` proves **NO moving cinematic footage** by itself.

### Still uncompleted / not claimed

- Verified fresh original LTX S02 33-frame GPU film sequence with natural bone jaw, neck/shoulder movement, consistent suit, and color-safe video.
- Seven independent original natively generated scenes, audio-conditioned human facial muscles and skeleton jaw synced to the six original Qwen Russian WAVs.
- 40+ seconds, >=576×1024, real independent QA and human artistic signoff, final MP4 GitHub link.

These require a truly completed GPU pilot first, GPU performance feasibility and human visual review. Not asserted solved by the CPU suite.

## Corrective evidence update — 2026-11-14 21:15 UTC

The earlier section describing the post-audit GPU pilot as **not executed** is now
superseded by this evidence. This is a significant correction to the incident
root cause and MUST be applied to follow-up automation:

- Native original S02 pilot **DID** execute on Tesla T4 and wrote a real
  `S02_PILOT33.mp4` before failing source-video QA, finishing at
  **329.805 seconds**. Source: the actual retained terminal GPU worker
  trace in the private Kaggle readback, safely archived by GitHub
  watchdog run `43634475448` artifact `14129681357`.
- Exact fatal traceback: **`RuntimeError: Authentic LTX S02 failed required
  video frame/color QA`**. The 1400 second inference timeout previously seen
  in other kernels did **NOT** recur in this third corrected run.
- Old watchdog incorrectly selected
  `UNSUPPORTED_CPU_HALF_PRECISION_VAE_OR_ENCODER` because it matched
  a nonfatal library warning **before examining the terminal receipt**. This
  was a misleading diagnosis, now corrected in
  `tools/eb01_gpu_watchdog.py` (receipt-first prioritization) and tested.
  CPU/FP16 warnings were present, but actual model synthesis completed.
- True reason for visual rejection is **not yet localized to frame geometry
  vs neon RGB**, because the original worker discarded frame-specific numeric
  QA and overwrote its receipt on error. This gap is repaired in future
  workers: `tools/kaggle_eb01_ltx2b_s02_fallback.py` now saves
  `EB01_S02_NATIVE_GPU_FRAME_QA_SAFE.json` with actual FFprobe clocks,
  relevant RGB suspicion frame indices, and exact reject checks, without
  weakening safety.
- The existing private GPU output is being examined, *without* resubmitting
  inference, by the new temporary-runner-only
  `tools/eb01_true_gpu_clip_numbers.py` and GH workflow
  `.github/workflows/eb01-private-actual-mp4-numerical-qa.yml`.
  One readback GH run `43641333448` was blocked by temporary
  Kaggle API **nonzero exit**, not by video inference. Server HTTP error
  class was not yet recoverable; do NOT infer exact 429/403 without proof.
- Kaggle API diagnostics now classify only explicit 429 (back off),
  401/403, 404, 5xx and connection categories, without printing any private
  original photos, six Qwen voices, token, or model prompts.
- Full actual suite **581 passed, 4 skipped**, run `43641583177`.
  This verifies CPU code/regression/privacy, **not cinematic quality**.
- Repo is PUBLIC, therefore never upload original image-derived
  `S02_PILOT33.mp4` to a GitHub Actions artifact, release, commit, issue, or
  externally visible file without owner approval. All diagnosis uploads
  are sanitized numerical JSON only. The source video remains private.
- Next: obtain original 33-frame numerical QA once Kaggle private output
  API is available; identify nonartistic geometry mismatch vs real neon
  contamination; solve actual root before launching a different GPU model;
  then apply full 7 distinct scenes/6 audio lip sync/40+ sec film QA.

## 2026-11-15 — Reanalysis of REAL neural GPU outputs and visual-quality RCA

The corrected short owner T4 GPU pilot in GitHub run [43643126165](https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/43643126165) produced an authentic 33-frame 320×576 24 FPS MP4 after **324.53 seconds** of native neural inference. The model **did not crash**. The independent color gate raised an exception after the MP4 had been written: all 33 frames had >0.15% "narrow neon" blue/magenta candidate pixels, only 4 had notable spectral change, **zero** frames had broad horizontal neon bars. Treating a nonfatal offload message about CPU float16 as the terminal failure was a watchdog classification defect corrected earlier.

The "narrow neon" detector historically tested **color coverage**, not line geometry, so an ordinary blue object visible throughout a scene could trigger it. The correction in `studio/eb01_motion_artifact_gate.py` now separately quantifies **1–6 row long chromatic line components**. The initial attempt incorrectly failed to detect a two-pixel stripe after FFmpeg's H264/YUV420 compression: decoded color spread to *four* rows. Reproduced in real CI with controlled 21-frame H264 videos. Root fix: group spatially coherent rows rather than using one-row neighbor isolation; blue areas spanning dozens of rows do not count as narrow lines. Constant cool-toned regions remain **REVIEW**, not automatically accepted footage. Both true thin RGB strips and broad blue scenic content therefore remain **blocked for independent cinematic assessment**. Targeted real FFmpeg QA [43658826151](https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/43658826151) passed 29 tests.

For the actual source GPU (not hypothetical synthetic footage), `tools/kaggle_eb01_ltx2b_s02_fallback.py` now preserves the private raw MP4, records original SHA and per-frame line geometry, and distinguishes "real model inference completed" from "aesthetic quality review blocked". `tools/eb01_gpu_watchdog.py` verifies 33 actual decoded frames/SHA in this intermediate case without calling a GPU-completed video a finished movie. The public GitHub workflow no longer writes private prompt-derived Kaggle output filenames into public action logs.

To investigate possible **numerical origin** of unusual RGB output, the next *single short* controlled A/B changes only the VAE encoder+decoder dtype to **FP32** while keeping transformer and T5 **FP16** (Tesla T4 lacks BF16). The model still sequentially offloads CPU/GPU and retains original source photos, seed, prompts, native 8-step scheduler and output geometry. A real reproducibility check against pinned upstream LTX-Video SHA `50f0aef1297574a4b29acbdb874e3a363a6d5f2e` compiled the exact patched official Python modules and verified 13 stage markers: [43658599778](https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/43658599778), 22 tests successful. This proves **patch compatibility**, NOT that FP32 VAE improves picture quality on live GPU.

The native 40+ second goal mathematically requires more than the 563-frame (23.458-second) V7 storyboard. V8 plan `studio/eb01_v8_40s_native_motion_plan.py` schedules 30 original independently generated 33-frame takes to assemble exactly **967 unique frames (40.292 seconds)** after trimming only 1 conditioning overlap frame on each continuation; no repeat/pseudo-frames. This is **only a plan**: real source character identity, six original Russian voice lines, audio-conditioned lipsync, motion anatomy, independent visual quality and owner artistic signoff remain unproven.

The GitHub repository is **public**, so raw private actor images, Qwen WAVs, video MP4 and full raw model logs must **NEVER** be uploaded to GitHub Actions artifacts, even for review; only numerically safe reports and preapproved code are allowed. Keep source originals within owner-private Kaggle dataset/notebook. Brother's separate Kaggle API authenticates, but original private source dataset ACL remains denied. Neither a finished 40+ second film nor its public-usable MP4 has been produced or published.
