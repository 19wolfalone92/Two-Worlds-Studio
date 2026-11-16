# Elizabeth Bones E01 — two independent Kaggle GPU workers (16 November 2026)

## Scope and truth boundary

This change introduces **working CPU-tested scene scheduling and read-only
connection checks**, not a completed film or permission to submit 30 expensive
GPU jobs. Existing main is untouched; six canonical Russian WAVs, original
portraits and V8 scene prompts remain SHA-locked and private. No TikTok posting.

**Existing verified video:** genuine 33-frame LTX 2B/T4 at 320×576, 24fps,
approximately 297 GPU seconds on one T4; motion is numerically near-static.
The earlier color pass does not prove acting, facial animation, bone-jaw
articulation, identity or lip-sync. The cinematic V8 target remains 30 unique
33-frame takes, 967 unique composite frames, >40 seconds and 576×1024 **native
detail**. Upscaling a 320×576 source cannot satisfy that requirement.

## GPU architecture: optimize reality, not advertised aggregate RAM

Kaggle's `NvidiaTeslaT4` accelerator corresponds to a **T4 ×2** session
according to Kaggle CLI maintainers. That is TWO **separate** cards (each
about 14.56 GiB visible in prior live model traces), not a single card
with pooled ~30 GiB. Two independently authorized Kaggle people could
each use an available session, giving at most four distinct T4 devices, if
Kaggle offers the sessions and their quotas allow it. None of this proves
four independent model processes will fit in CPU RAM or device memory.

Conservative runtime:
- Owner account: at most **one neural GPU process** by default.
- Collaborator account: at most **one neural GPU process** by default.
- Each account can opt into a *second* simultaneous process ONLY after its
  own T4×2 dual-process memory/time benchmark. Never set the flag because
  the Kaggle accelerator displays "x2".
- GPU work stays scene-sticky: no transfer of raw frames, WAVs, or secrets
  between accounts. Whole-scene exact assignment is **15 native takes
  for owner / 15 for collaborator**, assuming BOTH actually pass all gates.
  If collaborator is blocked, the owner alone gets the 30-scene proposal,
  not a fake "distributed" success.
- A single scene's successive 33-frame takes execute sequentially,
  sharing only the explicitly verified next-frame PNG inside that same
  account's own private storage. Multiple independent scenes may run
  concurrently after worker capacity proves safe.
- Always reserve 1.5× the independently measured native 33-frame time
  against the **individual account's reported remaining GPU-minute budget**.
  Never assume available weekly quota from an account token or from the
  other account's quota. Leave at least 0.75 GiB headroom per GPU beyond
  measured pilot peak. No cross-account GPU memory sum.

The 297-second **320×576** pilot is unsuitable for estimating runtime,
memory or picture quality at native **576×1024**. Both accounts must first
complete **their own native-resolution** measured short pilot with verified
VAE/transformer CPU offloading and a real human visual assessment. Existing
`studio/eb01_render_feasibility.py` rejects an unbenchmarked high-detail T4
profile using a conservative 24-GiB hardware floor. The new scheduler does
NOT bypass this production safety/quality gate or claim an untested T4 is
high-detail-ready. A separately reviewed, genuinely measured lower-VRAM
runtime would be needed before modifying that restriction.

## Independent account + privacy setup

Two *different authorized people* use their own Kaggle logins; never share
passwords or session cookies and do not create accounts to bypass quotas.
GitHub Actions secrets are `KAGGLE_API_TOKEN` for the owner and
`KAGGLE_API_TOKEN2` for the collaborator. The CI job passes the respective
tokens only to a read-only Kaggle command process.

Kaggle private dataset:
`wolf19loner92/eb01-v7-private-production-sources`

The dataset owner can grant the legitimate collaborator **Can view** using
Kaggle Dataset **Settings → Sharing**, without changing the dataset to
Public. Notebook sharing is **separate** from dataset sharing; individual
private notebooks require independent sharing approval. The collaborator
must have a successful authenticated metadata read before accessing
the source in an authorized private notebook.

Earlier independent Kaggle checks confirmed TWO distinct authenticated
accounts, but the collaborator lacked private dataset access. This cannot
be repaired by silently copying the original two portraits and six voice
files into public GitHub or by using the other person's credential.

Do **not** upload raw portraits, voice WAVs, generated MP4, source prompts,
or Kaggle download URLs to public GitHub Actions artifacts. Video handoff
requires private owner-controlled storage with verified ACL and SHA-256.
An ACL check that returns 401/403/404/429 or unclassified CLI failure
must remain blocked; never declare an absent dataset based on failure.

## What was actually added

- `studio/eb01_dual_kaggle_scheduler.py`: deterministic 7-scene/30-take
  allocation, whole-scene exact balancing, resume-after-verified-tail
  contract, in-flight locks, independent GPU/quota/VRAM constraints and
  two-T4 process-count benchmark gates.
- `tools/eb01_dual_kaggle_readiness.py`: read-only two-token authentication
  and metadata ACL check; returns sanitized status, never GPU media.
- `tests/test_eb01_dual_kaggle_scheduler.py` and
  `tests/test_eb01_dual_kaggle_readiness.py`: no-GPU regressions for
  account isolation, denied second source, scene continuity, quotas, VRAM,
  concurrency, and fail-closed missing reviewed native pilot.
- `.github/workflows/eb01-dual-kaggle-optimizer-ci.yml`: runs CPU-only
  tests and then a safe metadata-only Kaggle preflight under separate tokens.
  **It does not submit a single Kaggle job.**

The scheduler produces **proposals only**. Private workers must revalidate
source SHA, output-frame SHA, GPU identity, actual remaining quota, authentic
model execution, pixel content, color/temporal/actor QA, and the original
WAV before dispatching each take. A JSON receipt alone is never proof.

## Stepwise path to finishing E01

1. Check read-only report from the newly added GitHub Actions workflow.
   Missing/denied collaborator dataset access stays a hard blocker.
2. Authorize dataset collaborator read access using Kaggle UI; verify each
   account can see the same source dataset independently. Keep the dataset
   and notebooks private.
3. Select a genuinely T4-compatible native-detail model/profile using
   staged FP16 transformer and FP32 VAE, pinned commit, 8-step LTX2B
   configuration and memory-safe CPU offloading. Conduct ONE controlled
   GPU pilot at native >=576×1024 with 48–96 frames, without changing
   official model geometry or faking motion. If T4 OOMs, record peak RAM,
   last stage and exact reason. Do NOT repeat the same failing job blindly.
4. Verify actual decoded frames, full RGB corruption scan, real cheek/jaw
   and shoulder motion, source identity, and Russian speech alignment.
   Reject static/strange performance even if synthetic QA is green.
   Privately sign off first 2–4-second take only after human inspection.
5. Benchmark true per-device CUDA VRAM, the exact model's seconds/33 frames,
   per-account remaining GPU quota, storage size and two-GPU concurrency
   **separately**. Enable the second process only if each account's
   parallel benchmark does not exhaust host RAM or device VRAM.
6. Start scene-sticky 15/15 queues with fail-closed one-at-a-time
   checkpoints and no duplicate/retry storms. Check real Kaggle job states
   and preserve original output BEFORE numeric QA; quarantine any reject.
7. Assemble 30 authentic source takes (not the old 23-second rejected
   duplicate), 967 distinct full-resolution movie frames and six original
   Russian WAVs in owner-private storage. Independently review sync,
   natural motion, visual identity, subtitle timing and H264/AAC MP4.
   Deliver only a private owner review; **never auto-publish**.

## Acceptance criteria

- Two legitimate separate Kaggle accounts and source ACL verified, not
  merely two secret names.
- Live native 576×1024 high-detail pilot, correct SHA, 24fps 48–96 frames,
  real measured peak **per-GPU** VRAM and job duration for each worker.
- Natural cinematic performance + audio-conditioned jaw/face movement
  approved from the actual videos.
- Each source take has a unique SHA and verified same-account continuity,
  no duplicated source or missing frames, safe idempotent retries.
- 30/30 actual clips, 967 final distinct frames, six original WAV lines,
  full 40.292s decode and visual review, private durable MP4.
- No public source release, no TikTok publication and no main merge until
  those tests are actually passed.

References: https://www.kaggle.com/docs/datasets
https://www.kaggle.com/docs/notebooks
https://github.com/Kaggle/kaggle-cli/issues/1196
