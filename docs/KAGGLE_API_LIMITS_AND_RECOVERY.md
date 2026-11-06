# Kaggle API — verified limits and no-burst production policy

**Verified 2026-11-06 against Kaggle official API documentation.**

## Two distinct limits, not one number

- **GPU compute:** normally **30 GPU hours per 7 days** (weekly reset; sometimes
  higher depending on resource availability). Personal current hours must be
  queried separately using `kaggle quota --format json`. This is a compute
  balance and does NOT govern the number of API calls.
- **API request rate:** Kaggle officially states that API and website requests
  are subject to **dynamic rate limiting** and may return **HTTP 429**. Kaggle
  does NOT publish an exact universal fixed requests/minute or requests/day
  number. There is no trustworthy numeric "Kaggle API daily limit" to quote.
- **Model Proxy/Benchmarks AI inference quota:** a third, separate budget
  (`kaggle benchmarks quota`) with daily/monthly monetary-credit allowance.
  This is unrelated to GPU notebook hours or ordinary `kernels output`.
- Multiple access tokens of **one account** are not two independent GPU
  allowances and must not be used to sidestep API throttling.

Official references:
- https://www.kaggle.com/docs/api — Kaggle Public API / Rate Limits
- https://www.kaggle.com/docs/efficient-gpu-usage — weekly GPU hours
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md — output and status CLI commands
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md — separate Model Proxy quota

## What was costly in the earlier studio integration?

1. A redundant `kaggle quota` request occurred before each remote GPU job.
2. Each GPU notebook had a status poll every **45 seconds**; a 55-minute
   session could make **73–74 status requests** alone.
3. The ComfyUI readback on HTTP 429 could attempt **four** download
   commands with only minutes between them. CLI output may itself involve
   multiple metadata-page API requests.
4. One legacy portrait workflow downloaded *every* kernel output file, even
   large model weights, not just the two small proof files.

The studio now uses a shared Kaggle Actions concurrency group,
**3-minute** polling (~19 status checks over a 55-minute session),
no redundant preflight quota calls, **one filtered output attempt** with
max metadata page size 200, and a separate manually dispatched read-only
quota probe. There is deliberately **no automatic token-rotation,
rapid retry loop or rate-limit probing burst**.

## Interpreting a failure

- **429:** pause; do not re-run immediately. Kaggle recommends waiting
  several minutes and inspecting redundant calls. If persistent, pause
  longer and report reproducible feedback to Kaggle. There is no public
  fixed cool-down duration.
- **GPU remaining ~21.94h in the previous successful check:** historical
  value, not necessarily the current balance. A new successful read-only
  quota request is needed to update that figure.
- **429 does not imply no GPU hours**; likewise, remaining GPU hours do not
  guarantee API readback succeeds.
- **Readback success requires a real PNG/ZIP and receipt** verified against
  the exact run nonce. A green workflow that merely lists an available
  model/token does not prove completed generation.

## Privacy and durability boundaries

The source repository is PUBLIC. Generated story assets remain in a private
Kaggle notebook, not public GitHub Actions artifacts. The readback job can
verify the actual image only when Kaggle allows fetching it; it does not
publish, upload to TikTok, approve owner art or alter locked story scripts.

GitHub Actions `concurrency` serializes overlapping runs **within this
repository**, but cannot enforce a rate limit across other apps, manual API
clients, other repositories or the Kaggle website. No external provider
offers a guaranteed static limit; we avoid speculative counts.

## Safe one-time API diagnostic

Use **Actions → STORY GLITCH V3 - Kaggle API protection and quota → Run
workflow → check_quota=true**. It issues exactly ONE Kaggle CLI quota
command, does not use GPU or infer images, and prints a redacted report.
Alternatively run `python -m studio.kaggle_api_limits quota` from an
already authenticated trusted computer.

If Kaggle returns HTTP 429, the command fails immediately with a sanitized
message and does not automatically retry.
