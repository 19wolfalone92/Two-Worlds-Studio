# Hugging Face Hub → STORY GLITCH STUDIO

This integration verifies the GitHub Actions Hugging Face secret against the
official Hub and reads metadata for two approved open-weight models. It is a
**real connection preflight**, **not** an image/voice render or hosted-inference call.

## Activate

1. In Hugging Face Access Tokens, create a minimum-privilege fine-grained
   token for this integration (read-only access). Hosted inference permissions
   are unnecessary for this preflight.
2. In the studio GitHub repository, add an Actions repository secret named
   `HF_TOKEN`, not an ordinary repository variable. **Do not send it in chat**
   or commit it to the repo.
3. Review and merge this integration PR into `main`. Manual workflows must
   exist on the default branch to appear in GitHub's Actions UI.
4. Open **Actions → STORY GLITCH - Hugging Face connection check (read-only)
   → Run workflow**. It sends three read-only Hub metadata requests and
   reports only status, model revisions and declared license. It never
   prints an account name or the secret.

A green check proves that the token authenticated and both reviewed model
metadata records matched expectations. It does **not** prove that weights
can be downloaded, that Kaggle has a GPU, or that actual media was rendered.

## Models already referenced in our studio

| Purpose | Hugging Face model | License shown on model card | Compute |
| --- | --- | --- | --- |
| Image concepts and edits | `black-forest-labs/FLUX.2-klein-4B` | Apache-2.0 | GPU for real inference |
| English voice drafts | `hexgrad/Kokoro-82M` | Apache-2.0 | CPU/GPU |

The first is used by `studio/kaggle_flux.py`. The second is used through
`studio/kokoro_tts.py`. The metadata checker is implemented in
`studio/hf_hub.py` with standard-library HTTP only; it does **not** download
the checkpoints, load them or execute any model-supplied code. Runtime model
versions should be pinned and separately reviewed before production.

The existing production pipeline still requires human review of image
identity, audio quality, rights, subtitles and final video output.

## How Hugging Face, GitHub and Kaggle fit together

- **Hugging Face** supplies model repositories; free model weights are not
  unlimited free inference or unlimited cloud GPU.
- **GitHub Actions** runs unit tests, this authentication preflight and
  CPU-based editing. This check generates no images, no WAVs and no MP4s.
- **Kaggle** can run allowed open models when explicitly invoked using its own
  allocated compute. GPU availability, VRAM and quotas are not guaranteed.
- GitHub's `HF_TOKEN` secret is **not automatically available in Kaggle**.
  If a model needs authentication inside Kaggle, configure a separate private
  Kaggle Secret and retrieve it inside the private notebook/kernel; do not
  paste the GitHub token into Kaggle kernel metadata, source code or artifacts.
- Gated models may require accepting terms/permission separately. Metadata
  visibility is not evidence of permission to download gated weights.
- This workflow does not call Hugging Face's billed Inference Providers.

Safe no-network local preflight:

```bash
python -m studio.hf_hub --dry-run
python -m unittest discover -s tests -v
```

Official references:
- https://huggingface.co/docs/hub/security-tokens
- https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
- https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets
