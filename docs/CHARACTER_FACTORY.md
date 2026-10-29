# New characters from scratch, then views from their own generated identity

`studio.character_factory` is an execution layer in the existing Studio. Its
first job for each identity group uses **text only and no input artwork**. A
physical double waits for that new face and receives its own described body,
hair and costume. Later angles use that body's own generated front view. Old
Work art is not implicitly loaded or treated as a new design.

New designs are draft alternatives. They never overwrite a canonical reference
registry, replace a body in an episode, or grant art/rig/publication approval.

Reference-conditioned angle jobs explicitly request geometric reposing before
the identity description. A three-quarter task asks for a 45-degree turn of
the head and entire body, foreshortened shoulders and unequal cheek widths.
This addresses observed frontal copies in a real GPU batch; it is still a
prompt instruction, never an automatic visual-angle certificate. Check the
actual pixels and use the drawing desk's `request_revision` for concrete
defects without impersonating a human approval.

## Create a private new-design project

```json
{
  "schema": 1,
  "style": "Original hand-painted animation design with clear anatomy and expressive faces.",
  "bodies": {
    "actor_a": {
      "identity_version": "NEW_ACTOR_A_DRAFT_v001",
      "identity_group": "physical_doubles",
      "description": "Invent an original adult woman with the explicitly described physical appearance.",
      "anchors": {"hair": "Exact hairstyle and identifying detail side", "wardrobe": "The first body's costume"},
      "avoid": ["Text or logos"]
    },
    "actor_b": {
      "identity_version": "NEW_ACTOR_B_DRAFT_v001",
      "identity_group": "physical_doubles",
      "description": "The same new face with a distinct physical-body design and costume.",
      "anchors": {"hair": "The second body's hairstyle", "wardrobe": "The second body's costume"}
    }
  }
}
```

The first body in a group is its seed. Separate `identity_group` values create
independent faces. Body order is stored explicitly in the design fingerprint.
Image-reference fields in a bootstrap spec are refused instead of silently
reusing existing character images. Keep actual private designs outside Git.

```bash
python -m studio.character_factory create \
  --spec /PRIVATE/new_design.json --output /PRIVATE/NEW_DESIGN_v1
```

## Actual GPU execution

On a configured worker with CUDA and the existing FLUX dependencies:

```bash
python -m studio.character_factory run-gpu --project /PRIVATE/NEW_DESIGN_v1 \
  --max-images 6 --views front,three_quarter,profile --width 768 --height 1152
python -m studio.character_factory status --project /PRIVATE/NEW_DESIGN_v1
```

One lazily loaded `GPUImageSession` is shared by the whole sequential batch.
GPUs below 20 GiB use sequential CPU offloading and VAE tiling/slicing to keep
the requested native pixel dimensions. This can run slower; the real device,
offload mode and dimensions remain recorded instead of claiming a fast path.
Completed images are kept and validated before resumption. Re-running the
command generates only missing jobs; an incomplete working image is retained
for inspection instead of blindly overwriting it. Native output dimensions,
actual file decoding, backend/model/device provenance, exact prompt/input
fingerprints and generated-image hashes are recorded. A success message with
no real PNG, a blank image, changed identity bytes or conflicting completed
output cannot complete a job. The legacy single-image FLUX entrypoint uses the
same validated session interface.

The official checkpoint is `black-forest-labs/FLUX.2-klein-4B`, with its
documented four-step distilled sampling and CPU offloading. The model still
needs actual GPU memory and downloads. Planning or stub tests do not prove
quality, device compatibility or a successful inference run.

## Host-assisted drawing

The same ready job can be exported to an available drawing backend:

```bash
python -m studio.character_factory next --project /PRIVATE/NEW_DESIGN_v1 \
  --job actor_a__front --output /PRIVATE/ready_job.json
```

The host executes the exact prompt and the job's explicit reference list. An
initial seed has no input images; after it is imported, dependent jobs become
ready. Register the original generated PNG with the backend's real name and
an execution JSON containing `prompt_sha256` (the Studio fingerprint of
`{"prompt": "exact prompt"}`) and the ordered `reference_sha256` list.
Record unsupported controls truthfully, for example `seed_applied: false`
when the selected host tool does not expose a seed parameter.

```bash
python -m studio.character_factory register --project /PRIVATE/NEW_DESIGN_v1 \
  --job actor_a__front --job-file /PRIVATE/ready_job.json \
  --image /PRIVATE/actual_drawing.png --backend actual_backend_name \
  --execution /PRIVATE/actual_execution.json
```

`chatgpt_imagegen_host_bridge` means ChatGPT's built-in generator actually
produced the pixels from a Studio job. It **does not mean** the standalone
Python process ran GPU inference. `studio_flux_gpu` is used by the configured
GPU backend and records the actual model, device and precision.

## Return into the existing drawing/review desk

After every new body has its own generated front view:

```bash
python -m studio.character_factory export --project /PRIVATE/NEW_DESIGN_v1 \
  --output /PRIVATE/NEW_REFERENCE_PROJECT
python -m studio.character_art plan --project /PRIVATE/NEW_REFERENCE_PROJECT
```

This copies the real new image bytes into a separate unreviewed reference
project with original native dimensions. The normal drawing desk can then
create emotion, blink, hands and further camera-angle tasks. A reference sheet
is still not an aligned facial/puppet rig or an approved episode.

## Requested private Kaggle proof

`studio.kaggle_api prepare --mode design --design-spec ... --max-images 6`
bundles the reviewed factory source and text specification into a private
Kaggle worker. It never bundles API credentials, notebooks, old artwork or a
screenplay. The worker creates its own face reference, draws dependent views,
and returns nested PNGs, jobs and receipts. Output verification decodes the
real PNG/WAV bytes and checks the worker archive hash and run identity.

The `character-factory.yml` workflow is a bounded **user-requested** proof:
only the explicit `feat/from-scratch-character-factory` branch in the same
repository can invoke it by a PR request-file change; forks and ordinary CI
do not consume GPU. It can also be manually dispatched. The committed request
must change in the exact new source commit, not merely remain in the overall
PR diff. Later review-page or documentation commits therefore skip the GPU
job instead of repeating an old request. The committed request
contains anonymous technical character descriptions and a public recipient
key, not private production art or an authentication secret. The maximum is
six drawings. No schedule or publishing action exists.

The existing configured Kaggle connection submits the private worker. On the
public technical runner, returned art is authenticated-encrypted using a fresh
AES-GCM content key wrapped to the request's RSA public recipient key. Only
`TW_FACTORY_RESULT.twenc` is uploaded as a build artifact. Raw artwork and
the private recipient key are excluded; the private key remains in the
owner's separate recovery bundle. Wrong source/run context, tampering, and
overwriting an existing return file are rejected during local decryption.

This preserves private artwork while making a real Studio-generated result
retrievable for inspection. If credentials, GPU quota, downloads or inference
fail, the actual run must remain failed; a local/host-assisted image is never
substituted as proof of that GPU run.
