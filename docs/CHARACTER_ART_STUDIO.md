# Character Art Studio: current references, real drawings, inspectable variants

This is a drawing subsystem in the **existing** TWO WORLDS Studio. It addresses
three observed production problems: obsolete text descriptions overriding the
actual Work designs, generating a frontal character when the shot needs a back
view, and accepting technically readable files without inspecting anatomy/eyes.

It does not train a new neural network or guarantee facial similarity. Actual
pixels come from a real drawing backend. The repository supports the existing
user-operated ComfyUI and experimental Kaggle FLUX workers; ChatGPT can also
execute exported drawing jobs with its built-in image generator and import the
result. The built-in ChatGPT generator is **not embedded in the Python engine**.
No API key is needed for that ChatGPT-assisted workflow. Nothing downloads a
model, consumes a paid API, or transmits private references merely by planning.

## Private reference project

Keep all art projects outside this public repository. Use the current Work
handoff, not `story/visual_bible.json`'s old engineering-fixture descriptions.
The appearance specification has this shape (example is synthetic):

```json
{
  "schema": 1,
  "style": "Original carefully painted character illustration, restrained materials and clear shapes.",
  "bodies": {
    "earth_mira": {
      "identity_version": "EXACT_VERSION_FROM_HANDOFF_REGISTRY",
      "description": "Explicit physical appearance grounded in the current Work images.",
      "anchors": {"hair": "Exact current hairstyle and anatomical side of identifying details.", "wardrobe": "Exact current costume and materials."},
      "avoid": ["Obsolete costume from an earlier fixture"]
    }
  }
}
```

```bash
python -m studio.character_art init \
  --handoff /PRIVATE/WORK_HANDOFF --spec /PRIVATE/appearance.json \
  --output /PRIVATE/CHARACTER_ART_v2
python -m studio.character_art plan --project /PRIVATE/CHARACTER_ART_v2
```

Every imported reference must match the supplied registry's SHA-256 and body
version. Imported review status and original crop resolution are preserved;
a large upscaled thumbnail is never described as newly detailed original art.
Higher native-detail sources are preferred when choosing identity inputs.
No reference images means a failure, not a silently invented replacement face.

Nine drawing tasks per body: front, three-quarter, profile, back, rear
three-quarter, six emotions, six blink phases, hands, separated-parts plate.
The plan contains exact prompts, labelled physical-body references and hashes,
target dimensions, alpha requirements and unreviewed status. **Nine planned
tasks do not mean nine generated images.**

## Generate and retain candidates

For ChatGPT-assisted drawing: execute the saved job's prompt with its labelled
images, preserve the original generated PNG, record any execution additions and
actual backend, and import it. This import does not generate an image itself:

```bash
python -m studio.character_art register --project /PRIVATE/CHARACTER_ART_v2 \
  --job earth_mira__back --image /PRIVATE/actual_generated_back.png \
  --generator "actual drawing backend" \
  --note "Execution provenance, reference inputs and any prompt additions."
```

Candidates are keyed by image SHA, keep their source-job fingerprint, and are
never overwritten by another version. Reimporting the same bytes is idempotent
and preserves review. Altered references, recipes, or reviewed pixels fail the
next operation. Basic checks flag tiny files, blank images, unsuitable framing,
or fake transparency. These checks cannot certify faces, fingers, views, or
halo-free silhouettes. Transparent candidates are best inspected against both
light and dark backgrounds.

For a configured ComfyUI GPU with an installed reference-aware workflow:

```bash
python -m studio.character_art render-comfyui \
  --project /PRIVATE/CHARACTER_ART_v2 --job earth_mira__profile \
  --workflow /PRIVATE/installed_reference_workflow.json \
  --checkpoint installed_model.safetensors --url http://127.0.0.1:8188
```

The `__REFERENCE_IMAGE__` LoadImage node must reach the **active sampler** that
feeds SaveImage node `9`. An unused uploaded reference is rejected before an
upload or render. This is a graph check, not proof that a particular model or
adapter is suitable. The backend currently uses the highest-priority single
identity reference; multi-body jobs and targeted edits require a separate
explicitly configured backend. No default text-only workflow is substituted.

The existing optional FLUX worker can use the same current character project:

```bash
python -m studio.kaggle_flux --mode character \
  --character-project /PRIVATE/CHARACTER_ART_v2 --job earth_mira__profile \
  --output-dir /PRIVATE/flux --dry-run
```

Remove `--dry-run` only on a configured GPU worker. GPU memory, model downloads,
and generation quality still require actual execution; a successful CPU plan
does not verify them. FLUX currently uses one identity reference and its own
resolution limits. Reference hashes are checked again before inference.

## Actual screenplay and camera staging

The art director now accepts `--character-project` and the private handoff's
`--assets`. This overrides old fixture descriptions without editing the
screenplay. It works for both character shots and environment-only shots:

```bash
python -m studio.art_director --episode /PRIVATE/WORK_EPISODE.json \
  --assets /PRIVATE/WORK_HANDOFF/assets \
  --character-project /PRIVATE/CHARACTER_ART_v2 --output-dir /PRIVATE/shot_art
```

An individual Work scene can also become a drawing job with optional explicit
camera staging. Staging accepts only `view`, `facing`, and `plane` for bodies
already in the scene, so the studio does not add characters or rewrite action.

```bash
python -m studio.character_art scene --project /PRIVATE/CHARACTER_ART_v2 \
  --episode /PRIVATE/WORK_EPISODE.json --scene S03 \
  --staging /PRIVATE/camera_staging.json --output /PRIVATE/scene_job.json
```

Example placement: `{"earth_mira":{"view":"back","facing":"mirror","plane":"foreground"}}`.
The source scene/episode hashes and staging are retained in the drawing job.
A different staging under the same existing job ID is refused; retain a new
job version instead of silently replacing the old candidate's source contract.

## Compare, correct and inspect coverage

```bash
python -m studio.character_art gallery --project /PRIVATE/CHARACTER_ART_v2 \
  --output /PRIVATE/CHARACTER_REVIEW.html
python -m studio.character_art status --project /PRIVATE/CHARACTER_ART_v2
```

The self-contained Russian HTML report embeds private drawings and original
references, filters by body/status, opens large previews and exports review
drafts. It works without an external server, third-party fonts, analytics, or
image hosts. Seven separate review criteria: identity, anatomy, physical
left/right details, wardrobe, pose, eyes, and style. The report does not send
reviews anywhere or approve its own results.

After an actual human has supplied a review export:

```bash
python -m studio.character_art review-batch --project /PRIVATE/CHARACTER_ART_v2 \
  --review /PRIVATE/character_review_drafts.json
python -m studio.character_art revision --project /PRIVATE/CHARACTER_ART_v2 \
  --job earth_mira__back --candidate EXACT_CANDIDATE_ID \
  --output /PRIVATE/correct_back.json
```

The entire review batch is validated before any review is written. Acceptance
requires all seven explicit visual assessments, no technical findings, an
identified reviewer, notes and the exact reviewed image hash. A revision keeps
the identity references, source drawing, concrete corrective prompt and lineage;
its new candidate starts unreviewed. Replanning preserves these additional jobs.

**A drawing, even an accepted one, is not a finished animated actor.** A parts
plate needs alignment, hidden joint paint, pivots, hierarchy, facial replacement
sprites and authored tracks before `CharacterRig`. A six-panel blink sheet is
not an aligned eyelid overlay or a replacement for the existing erase-mask gate.
Neither drawing review nor technical PASS grants episode/publication approval.

## Verification

`tests/test_character_art.py` uses synthetic art only: reference version/hash
drift, original crop disclosure, changed jobs, immutable/repeated imports, real
alpha, review-to-pixel binding, revision lineage, camera staging without script
edits, offline gallery escaping, existing art-director/FLUX integration, and
disconnected reference workflows. Private production images stay out of CI and
the public repository.
