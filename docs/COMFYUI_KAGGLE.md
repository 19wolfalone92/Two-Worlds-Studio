# TWO WORLDS — ComfyUI over Kaggle API (private one-shot proof)

## Status / scope
This adapter runs the existing `studio.art_director` ComfyUI integration
against a *real local ComfyUI server inside a Kaggle T4 session*. It does NOT
provide a permanent hosted GPU, remote public endpoint, rendered ARCHIVE 17
scenes, facial reference control, voiceovers, MP4 or automatic publication.

The first job deliberately renders only scene S01 of the already-public
`episodes/E01_THE_REFLECTION.json` engineering fixture, with base SDXL.
The production canon is controlled by Work and is not bundled or altered.

## Security and cost
- Trigger is opt-in by a single change to `ops/comfy_launch.trigger` in
  feature branch `feat/comfyui-kaggle-api-20261106`, or workflow_dispatch.
  Ordinary code/PR changes do NOT consume GPU quota.
- GitHub Actions reads `KAGGLE_API_TOKEN` secret and uses
  `studio.kaggle_identity` to resolve Kaggle username.
- Only reviewed public source files, the public E01 fixture and the ComfyUI
  API template are embedded in the **private Kaggle kernel**. Secrets,
  unpublished Work scenarios, reference portraits and private files are not.
- The model checkpoint is downloaded **inside Kaggle**, not into GitHub.
  Review model licenses (SDXL OpenRAIL++-M) and Kaggle quota requirements.
- No tunnels are opened. ComfyUI listens on `127.0.0.1:8188` only.
- This public GitHub repository never uploads the resulting images to public
  GitHub Actions artifacts. Results remain in the **private Kaggle notebook**
  owned by the account.
- No TikTok/Metricool publication and no auto approval of generated art.

## Files
- `studio/comfy_kaggle.py`: prepares a private Kaggle job and verifies the
  returned ZIP/receipt, PNG signature, dimensions, ComfyUI prompt id and SHA.
- `tools/comfy_kaggle_entry.py`: installs ComfyUI, downloads the publicly
  available SDXL checkpoint, launches local API, submits one shot using
  `studio.art_director.comfy_render`, and packs a real PNG/manifest.
- `.github/workflows/comfyui-kaggle.yml`: once-only opt-in GPU workflow.
- `tests/test_comfy_kaggle.py`: CPU-only packaging/security/receipt checks.
- `templates/comfyui_sdxl_api.json`: existing reviewed node graph.

## What counts as verified
1. GitHub CI offline packaging and tests pass.
2. Kaggle accepted the job AND reported a completed GPU kernel.
3. Runner downloaded ZIP + receipt; verifier finds an actual PNG with expected
   640×1152 dimensions, matching SHA and a real `comfy_prompt_id`.
4. A human verifies visible image anatomy, quality, rights and likeness before
   any master shot goes to production.

Any quota, download, ComfyUI startup, workflow compatibility, sampler, or memory
error must be reported as a failure; it cannot be silently replaced with
procedural fake art. Consistent faces will require owner-approved references
and a real IP-Adapter or equivalent conditioning workflow: this SDXL smoke
test does not implement or claim that control.

## Local offline check
```bash
python -m pip install -r requirements.txt
python -m unittest tests.test_comfy_kaggle -v
python -m studio.comfy_kaggle prepare \
  --username local-example --nonce local-smoke-001 \
  --output-dir /tmp/two-worlds-comfy
```

## Actual GPU results
Private Kaggle URL after successful API launch:
`https://www.kaggle.com/code/<authenticated-username>/two-worlds-comfy-sdxl`.
The image is **not** pushed to GitHub. A GitHub workflow success without the
real PNG verifier is not accepted as generation success.

## Future production handoff
Once this smoke test works, use locked Work scene manifests, approved source
portraits with reference conditioning, batch generation and `studio.asset_ingest`
to move *reviewed* visuals into the existing hybrid MP4 assembly pipeline.

## Reference conditioning proof (code-level, not a likeness guarantee)

`studio.comfy_reference_guard` now rejects an unused reference image:
an approved `LoadImage.image = "__REFERENCE_IMAGE__"` must feed an enabled
`IPAdapterAdvanced` / supported IP-Adapter *application* node, with a
nonzero weight and a connected `ipadapter` weights input. Its conditioned
model must feed the `KSampler` that actually flows through `VAEDecode` to
`SaveImage` node 9. Simple reference uploading, unconnected LoadImage
nodes, an adapter with weight 0, or a sampler using the unconditioned
checkpoint fail closed **before uploading owner-approved character art**.

To use reference conditioning in real GPU production, a tested custom
ComfyUI graph, compatible IP-Adapter nodes and properly licensed/pinned
weights still need to be installed on the actual Kaggle GPU runtime.
This check proves only *graph topology*; it does **not** prove custom
nodes installed, model weight availability, human/fictional-face identity,
or image quality. The existing unconditioned SDXL smoke-test continues
to run with its original workflow.
