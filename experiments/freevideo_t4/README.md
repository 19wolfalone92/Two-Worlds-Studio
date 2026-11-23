# Freevideo — isolated T4 experiment

**Status (23 Nov 2026): independent research prototype, not merged.**
This is the temporary **Freevideo** project name to distinguish it from the
production **Two Worlds Studio**. The code lives only on
`research/freevideo-t4-adaptive-20261123`, under
`experiments/freevideo_t4/`.

**What it IS:** a memory-aware *heuristic* and two-stage, operator-reviewed Wan2.2
image-to-video workflow that delegates inference to the existing
`studio.comfy_neural` bridge. GPU-specific plans, provenance checks, telemetry,
local ComfyUI device identity consistency, and fail-closed rules are included.

**What it IS NOT:** a merged neural model, MiniMax H3 implementation, Video
DeltaNet model, real four-GPU scheduler, actual measured performance improvement,
real-world animation QA, or a production release. No FreeVideo project source
code/model weights are incorporated, and the project's separate license
conditions remain unaffected. The project name is a label, not a claim of
technical equivalence.

## Audit findings and fixes

| Chain | Previously | After audit |
| --- | --- | --- |
| Device selection | Heuristic trusted a local CUDA VRAM query without verifying ComfyUI's GPU | Actual T4 required; `/system_stats` must report matching device index/type/name/total VRAM |
| Memory | No invalid-VRAM validation; could allow nonsensical reports | Reject NaN, over-total and insufficient VRAM; no implicit OOM retries |
| Preview → final | A manually written review JSON and arbitrary video hash could authorize final | Source, preview MP4, native Studio receipt, Freevideo stage receipt, prompt, seed and frame clock must match |
| Output provenance | No Freevideo-specific take receipt | Private SHA-256-bound `.freevideo.json` stage receipt with hardware, parameters and bridge-receipt digest |
| Frame/geometry | Not independently checked at the Freevideo boundary | `ffprobe` counts frames and verifies width, height and 24 FPS |
| File safety | Symlinks were resolved before they could be rejected; public-path alias risk | Check lexical and resolved paths; deny file symlinks, never overwrite outputs or receipts |
| Telemetry | Malformed/NaN JSONL could crash or mislead stats | Bounded streaming reader, strict numeric validation, real-run medians only |
| Concurrency | Two tasks could race for one T4 and waste GPU quota | Per-worker per-device exclusive file lock, no implicit scheduling across Kaggle sessions |\n| Recovery | A telemetry write failure could look like failed GPU inference | Keep an already verified MP4 and return a clear telemetry warning; no blind automatic reruns |
| Quality | A JSON `approved` flag could be misread as objective quality evidence | Human approval is an explicit manual assertion, not a computer vision score |

**Important:** this branch has no independent motion/identity recognition model.
A person must inspect full video for face consistency, hands, rigid skull and jaw,
warping, temporal flicker and natural physical movement. The reviewer must never
set any checks to `true` without actually watching the clip.

## Offline regression tests

From the repository root with **Python 3.11+** and FFmpeg/ffprobe installed:

```bash
python -m unittest discover -s experiments/freevideo_t4 -p 'test_*.py' -v
python experiments/freevideo_t4/adaptive_pipeline.py plan --simulate-t4
```

The isolated tests **do not require CUDA or Kaggle**; only an FFmpeg CPU fixture
decodes a real synthetic 25-frame MP4. GPU inference is mocked in the tests.

Prior 23 Nov audit reported **23/23 offline tests passing**. Additional integrity, source-tamper, and CPU temporal-QC tests have since been added; use the latest completed branch GitHub Actions run for the authoritative updated count. This proves
software logic only, not T4 inference or film quality. GitHub Actions' independent
result is not assumed without its own completed check.

## Optional real manual experiment

Before any real generation: install legitimate Wan2.2 TI2V-5B weights in a
private ComfyUI, run it on the **same CUDA T4** as this CLI, with ComfyUI bound to
localhost and `/system_stats` reporting the selected GPU index. Ensure the
GPU has enough free memory, PyTorch/FFmpeg are installed, the reference is
approved, and model licenses permit the intended use.

The CLI probes just **one CUDA GPU**, and forbids use of a different model or
GPU. 2 T4s per Kaggle session are separate devices; two Kaggle sessions do
not form one 64-GB GPU. Do **not** call four GPUs a combined model.

Do not put reference art, model weights, generated MP4, human review JSON,
receipts or GPU logs into this PUBLIC repository.

```bash
# Simulation only — no GPU, no model load, no network calls:
python experiments/freevideo_t4/adaptive_pipeline.py plan --simulate-t4

# On a real Kaggle T4 with compatible private local ComfyUI:
python experiments/freevideo_t4/adaptive_pipeline.py plan

# Explicit GPU preview (25 frames, ~1 second). Never automatic.
python experiments/freevideo_t4/adaptive_pipeline.py preview \
  --reference /private/eb01/shot.png --output /private/eb01/preview.mp4 \
  --weights-manifest /private/eb01/wan-weights.json \
  --prompt 'A skeletal character turns head and shoulders while rigid skull bones preserve their exact shape.' \
  --seed 42 --history /private/eb01/timing.jsonl \
  --accept-model-license --execute
```

The successful preview produces:
- `preview.mp4` — native video marked **REVIEW REQUIRED**;
- `preview.receipt.json` — Studio bridge provenance;
- `preview.freevideo.json` — extra Freevideo stage provenance.

**Manually inspect the actual preview**. Create the following external
`/private/eb01/review.json`, filling SHA values from the real files and
setting all checks true **only if genuinely satisfied**:

```json
{
  "schema": 1, "approved": true, "reviewer": "owner",
  "preview_sha256": "<SHA256 preview.mp4>",
  "reference_sha256": "<SHA256 shot.png>",
  "checks": {
    "identity": true, "anatomy": true, "motion": true, "no_morphing": true
  }
}
```

Then, deliberately, with the **same** prompt and seed:

```bash
python experiments/freevideo_t4/adaptive_pipeline.py final \
  --reference /private/eb01/shot.png --output /private/eb01/final.mp4 \
  --weights-manifest /private/eb01/wan-weights.json \
  --preview-video /private/eb01/preview.mp4 --review /private/eb01/review.json \
  --prompt 'A skeletal character turns head and shoulders while rigid skull bones preserve their exact shape.' \
  --seed 42 --history /private/eb01/timing.jsonl \
  --accept-model-license --execute

python experiments/freevideo_t4/adaptive_pipeline.py stats --history /private/eb01/timing.jsonl
```

**Even after passing, `final.mp4` is a new independent generation** — preview
approval is not approval of the final take. Its artistic quality must be
visually examined again using the existing studio's human release policy.


## Mandatory private weight-integrity manifest

Every real `preview` or `final` call now requires `--weights-manifest`.
The following JSON is a **schema example only**: use exact absolute paths and
trusted expected SHA-256 digests for the three checkpoint files actually
installed in the local ComfyUI directories. Do not commit manifests or weights.

```json
{
  "schema": 1,
  "model": "wan22-5b",
  "license_reviewed": true,
  "source_repository": "https://huggingface.co/Comfy-Org/",
  "files": {
    "wan2.2_ti2v_5B_fp16.safetensors": {
      "path": "/private/comfy/models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
      "sha256": "<64 lowercase hex characters from a trusted expected digest>"
    },
    "wan2.2_vae.safetensors": {
      "path": "/private/comfy/models/vae/wan2.2_vae.safetensors",
      "sha256": "<64 lowercase hex characters from a trusted expected digest>"
    },
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors": {
      "path": "/private/comfy/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
      "sha256": "<64 lowercase hex characters from a trusted expected digest>"
    }
  }
}
```

```bash
python experiments/freevideo_t4/weights_preflight.py /private/eb01/wan-weights.json
```

This checks local bytes against operator-supplied expectations. It **does not**
authenticate the original publisher, prove which weights a running ComfyUI
process loaded, or demonstrate T4 compatibility. The FP8 text-encoder storage
filename does **not** imply that the NVIDIA T4 has FP8 Tensor Core arithmetic.
A documented model licence review remains necessary.

## Independent CPU temporal diagnostics

```bash
python experiments/freevideo_t4/quality_metrics.py /private/eb01/preview.mp4
```

This performs strict full-frame FFmpeg decoding (a separate CPU test) and
reports near-duplicate adjacent frames and abrupt pixel changes. It never
treats camera motion, image panning, or moving textures as proof of genuine
character articulation. Nor does it verify neural provenance or the likeness
of Elizabeth, the rigid skull of Kostya, or artistic suitability. These are
strictly `NOT_VERIFIED` until examined on authentic generated footage.

## Unknown-running GPU job quarantine

If the Studio ComfyUI subprocess times out, returns an error, or produces
malformed JSON, Freevideo leaves a persistent private `gpu-INDEX.quarantine.json`
marker under its per-worker lock folder. Subsequent Freevideo launches on that
device fail with `GPU_QUARANTINED`, **even after the original subprocess exits**.
This deliberately prevents a second expensive GPU submission while ComfyUI
might still be working on the first prompt.

**Manual recovery procedure**: investigate `/queue` and `/history` on the
same trusted local ComfyUI instance, inspect generated MP4/receipt paths and
any incomplete jobs, ensure the GPU is fully idle, then have the responsible
operator explicitly remove the quarantine marker before any new render. Do not
remove it automatically or as a reaction to a transient API error. This marker
is **per Kaggle worker** and does not coordinate separate Kaggle sessions.

The CPU regression explicitly simulates the timeout condition. It does not
prove handling of a real remote ComfyUI process crash, GPU reset, or Kaggle
worker restart (worker-local state may be lost on restart).

## Gates before considering selective integration into main

1. Verified *real T4* on the actual Kaggle worker. No simulation may count.
2. Real ComfyUI backend receipt and VRAM/temperature/time observations.
3. Visual comparison of Elizabeth and Kostya: shape, anatomy, identity, motion,
   texture stability and absence of floating/deforming objects.
4. Calculate **end-to-end GPU seconds per usable approved final video** against
   a baseline. A low-quality preview can still cost more, not less.
5. No unexpected downloads, credentials exposure, cost, or auto-publication.
6. Independent owner review and separate explicit approval before porting **only
   proven improvements**. **Never merge this experiment automatically.**

Upstream conceptual references: [FreeVideo](https://github.com/FlashML-org/FreeVideo),
[Video DeltaNet](https://arxiv.org/abs/2609.20744),
[Wan2.2 ComfyUI](https://docs.comfy.org/tutorials/video/wan/wan2_2).
