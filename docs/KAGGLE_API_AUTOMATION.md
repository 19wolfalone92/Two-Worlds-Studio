# Kaggle API → GitHub Actions: one-time private setup

This is the **API route**. No manual Kaggle Notebook editing, no API keys in
ChatGPT messages, no running Python commands on a phone.

## One-time account setup (owner only)

1. Sign into Kaggle and open https://www.kaggle.com/settings/api .
   Under API Tokens create a new **personal API token** for automated CLI use.
   Copy it in your browser. Do **not** post it in chats, repository code or
   GitHub issues. Kaggle also supports legacy kaggle.json auth, but this
   workflow uses the current KAGGLE_API_TOKEN environment variable method.
2. Open the GitHub project
   https://github.com/19wolfalone92/Two-Worlds-Studio/settings/secrets/actions
   and choose New repository secret. Set its name exactly to
   **KAGGLE_API_TOKEN**, and paste your Kaggle token in the Value field.
   GitHub encrypts repository secrets and masks values in standard logs.
3. In the same page, open the **Variables** tab, create variable
   **KAGGLE_USERNAME**, value equal to your exact Kaggle username
   (not email). This is an ordinary non-secret variable.
4. **Review and merge the PR stack** before using the Actions UI.
   GitHub's Run workflow button appears only when the workflow is available
   on the DEFAULT branch. No merge has been made by this setup.
5. GitHub → Actions → **TWO WORLDS - Manual Kaggle API GPU** →
   **Run workflow**. Choose mode 'portrait', body 'earth_mira', view 'front'.
   This is an explicit, potentially quota-consuming GPU run you control.
6. GitHub sends a private Kaggle kernel with the official CLI,
   requesting the available T4 GPU, then watches until it completes
   and validates run ID, image/audio existence and archive integrity.
   External free-GPU access or successful model execution is not guaranteed.
7. Results: for a PRIVATE GitHub repo the ZIP is also a GitHub Actions
   artifact (3-day retention). For the current PUBLIC repo, the workflow
   intentionally DOES NOT copy unpublished art/audio into public GitHub
   artifacts. Download from the private Kaggle kernel instead.

## Security and design

- Token remains only in GitHub encrypted Actions repository secrets.
- Kaggle receives a small reviewed bundle of source modules and metadata,
  not GitHub credentials, scripts, account data or unpublished Work text.
- The kernel is configured private. No publishing or upload to TikTok.
- Only workflow_dispatch initiates GPU consumption. No recurrent automation.
- A unique run nonce prevents accepting stale/another job's files.
- Actual images or WAVs are UNREVIEWED until genuine owner approval.
- If the model runs out of memory or Kaggle quota, workflow reports failure.

## Components

- studio/kaggle_api.py: generate / monitor / validate.
- tools/kaggle_api_kernel_entry.py: Kaggle side kernel template.
- .github/workflows/kaggle-api.yml: manual trigger with secured token.
- studio/kaggle_flux.py: FLUX.2 Klein 4B portrait generator.
- studio/kokoro_tts.py: Kokoro-82M synthetic English sample generator.

Modes:
- portrait: first 576×1024 concept portrait PNG for a character;
- voice: short Kokoro WAV sample; CPU is supported;
- both: both, sequentially, on the same Kaggle session.

Only after actual generated outputs and approvals should the motion-comic
producer ingest them into production. Work remains the exclusive script writer.

## What is verified today

The PR's GitHub CI can package the kernel and run CPU-only unit tests.
No Kaggle token exists in CI by default. A green test is NOT proof that
Kaggle allocated a GPU, downloaded FLUX weights or produced professional art.
Only a real API run after owner token setup can verify this.

## Official documentation
- Auth: https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md
- Metadata: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
- Push/status/output: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
- GitHub workflow dispatch:
  https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow

## Optional username discovery after live connection check

The initial connection-check run found an encrypted Kaggle API token but no
GitHub repository variable named KAGGLE_USERNAME. To reduce the work required
from the owner, the new workflows first authenticate through a read-only
`kaggle quota` call, then try resolving the account handle using the
authenticated `kaggle config view`. The result is validated and passed
through GitHub Actions environment without printing secrets. If the
token-backed CLI cannot resolve the username, the owner still needs to
create variable KAGGLE_USERNAME under the repository's Actions Variables.
This preflight does not request a GPU.
