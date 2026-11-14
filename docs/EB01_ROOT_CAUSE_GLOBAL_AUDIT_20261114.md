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
