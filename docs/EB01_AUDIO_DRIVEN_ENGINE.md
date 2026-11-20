# Elizabeth Bones: audio-driven actor experiment

The established LTX pilots generated real 576×1024 pixels but moved very
little and did not condition the model on the Russian voice. This experiment
adds a separate EchoMimicV3-Flash path while preserving that working fallback.

## Immutable inputs and neural source

- Branch: `feature/eb01-audio-driven-actors-20261120`; draft PR #74.
- Upstream: `https://github.com/antgroup/echomimic_v3.git`.
- Pinned upstream: `7e89489ca51c0d008fc1963ec6c03fc5bd0b9397`.
- Original portraits and all six original WAVs are SHA-locked by
  `studio.eb01_v7_storyboard`; the shared dataset stays private.
- Public model revisions and every selected file hash are recorded by the
  actual CPU preflight and verified again after downloading on Kaggle.
- No synthesized voice, replacement portrait, face patch, frame interpolation,
  output upscale, per-frame color transfer, paid GPU or social publication.

## Device placement and memory

Each account runs one video process using its two separate T4 devices.
GPU 1 holds the FP16 T5 encoder. GPU 0 stages the FP32 video VAE, FP16 CLIP
and FP16 Flash transformer in sequence. VRAM is never pooled. Models remain
on CPU between their stages. Padded SDPA replaces unsupported FlashAttention
with a real mathematical equivalence test before GPU submission.

The T5 file is downloaded and hash-verified first. After T5 is loaded on
GPU 1, its temporary disk copy is released before downloading vision weights.
This avoids keeping all ~20 GiB of model files together on a limited scratch
volume. The actual GPU process samples RAM, CPU, disk and both individual
VRAM peaks. The native denoised latents are saved before decoding.

## Pilot clocks

| Account alias | Actor / shot | Native geometry | Native frame clock | Original speech |
|---|---|---|---|---|
| owner | Kostya / S06 | 576×1024 | 89 frames, 25 FPS, 3.56 s | Full original WAV, 3.28 s |
| collaborator | Elizabeth / S01 | 576×1024 | 65 frames, 25 FPS, 2.60 s | Full original WAV, 2.16 s |

These are different jobs, not duplicated scenes. The entire WAV is resampled
to the acoustic encoder clock and padded with silence. Duration rounds UP
to the model's `4n+1` frame constraint. Original speech is never cut. The
original WAV is also muxed into the final MP4; a soundtrack alone is not
accepted as evidence of audio conditioning.

## Real gates and private recovery

`eb01-echo-two-account-pilot.yml` runs actual neural CPU imports, checks SDPA
padding against explicit attention math, checks every pinned model URL, and
requires both independent accounts to download all eight original hashes.
Only then may it submit one private notebook per account.

A kernel name is deterministic from run ID and account. A workflow retry
only reads/collects that exact notebook, even after an uncertain push.
The two account concurrency groups also serialize other studio GPU work.
There is no thirty-take dispatcher in this experiment.

`eb01-echo-collect.yml` reads `ops/eb01_echo_collect.json` and passes an
explicit original run AND source revision. This path never submits GPU work.
A valid RUNNING/QUEUED observation succeeds as a status read, with quality
approval still false. A terminal failed pilot remains failed.

Independent CPU acceptance fully decodes actual RGB frames and audio;
checks 576×1024, the exact 25 FPS / 65-or-89-frame clock, distinct decoded
frame hashes, no neon, the full voice track, source revision and the actual
model audio-conditioning receipt. It accepts the audio engine's explicit
25 FPS without weakening the default 24 FPS fallback contract.

Raw MP4, source-sensitive diagnostics and the denoised latent checkpoint are
returned only in an authenticated RSA-3072 / AES-GCM envelope. The runner
has only the public return key; the private key is outside the repository.
Public artifacts contain bounded numerical evidence and ciphertext only.

## Artistic acceptance

A technical pass is not approval of acting. The actual MP4 must be reviewed
for Russian articulation, jaw/chin, head/neck/shoulder/body motion, identity
preservation, flicker and deformation. In particular, animation of the skeleton
must not acquire human facial tissue. Frame uniqueness and motion deltas
cannot prove any of these artistic properties. Production and main merge stay
blocked until a successful pilot and explicit quality confirmation.

## Verified runtime incident and correction

The first Echo attempt (`44817318833`) failed before neural model loading:
Kaggle's actual interpreter was Python 3.13, and the original SentencePiece
pin attempted an unsupported source build. Both private setup logs confirm
that same cause. A Kaggle COMPLETE state therefore does not prove a video.

The runtime now uses NumPy 2.1.3 and SentencePiece 0.2.1, with binary wheels
required for native libraries. CPU preflight runs Python 3.13 and PyTorch
2.6; it also downloads/hash-checks the actual public pretrained acoustic
encoder and performs a real 24 kHz → 16 kHz → 25-frame forward. Both CPU
preflights `44818627372` and `44818907308` passed before the corrected GPU
attempt. The renderer and RGB artifact decoder close process pipes on both
success and error, including batch rendering.
