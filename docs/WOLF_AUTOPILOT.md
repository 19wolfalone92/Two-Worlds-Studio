# WOLF AI — autonomous auditor (phase 1)

WOLF AI is an **actual runnable, read-only MVP**, designed to conserve ChatGPT Work allowance by moving deterministic audits into GitHub Actions or a local machine. It **does not** unlock Work usage, pretend to contain a free unlimited LLM, author scripts, place trades, commit to another repository, or publish videos.

## Current scope

The allowlist includes `studio` (Two-Worlds-Studio), `shaira`, `williams`, `williams-ai`, `english-child`. Commands are explicit, bounded and branch-safe. `studio` includes a readiness command and selected tests; `shaira` runs its declared npm check/test script; the other projects receive **syntax/inventory checks only** until individually approved CI contracts are defined. The Williams Bot implementation must remain in its dedicated development chat; this workflow **never modifies that bot**.

## Quick start

From the Two-Worlds-Studio checkout:

~~~bash
python -m unittest discover -s tests -p 'test_wolf*.py' -v
python -m wolf_ai.orchestrator --project studio --mode audit --source . \
  --output ../wolf-audit.json
# Explicitly opt into running checks:
python -m wolf_ai.orchestrator --project studio --mode audit --source . \
  --output ../wolf-audit.json --execute
~~~

For GitHub: after merging the PR, open **Actions → WOLF AI — free-first project audit → Run workflow**, choose the project and mode. A weekly *studio-only* audit is scheduled. Results are found in the `wolf-audit-<run-id>` workflow artifact; no repo changes are made.

## Optional local AI adviser

An OpenAI-compatible endpoint is **optional** and is disabled unless `--send` is set. The adviser sees **only status codes**, *not source code, voice files, scripts, test logs or secrets*, and its response is informational (no edits). For Ollama on the user's PC:

~~~bash
ollama pull qwen2.5-coder:7b
python -m wolf_ai.ai_review --report ../wolf-audit.json \
  --output ../wolf-advice.json --send \
  --base-url http://127.0.0.1:11434/v1 --model qwen2.5-coder:7b
~~~

Other providers require a user-provided endpoint/model/token and potentially incur costs. Never enter provider credentials into a public repository. Do not run optional paid cloud inference automatically.

## Optional bounded AI patch proposal

The source-to-diff tool is **off by default**. It may transmit up to 3 explicitly
selected UTF-8 source files to your selected model when you pass
`--send-source`. **Do not select any files containing private information,
credentials, or licensed scripts.** A local Ollama endpoint is one option.
A paid remote endpoint may incur charges.

From the Studio checkout, assuming Ollama and the model are already running:

~~~bash
python -m wolf_ai.patch_proposal --source . \
  --file studio/smooth_camera.py \
  --issue "Propose a narrowly scoped improvement to camera motion smoothing" \
  --output ../wolf-camera-proposal.patch \
  --send-source --base-url http://127.0.0.1:11434/v1 \
  --model qwen2.5-coder:7b
~~~

The model's output must be a unified git diff touching only the selected files.
The tool validates the paths and uses `git apply --check`. It **does not**
apply the patch, execute proposed changes, open a PR, or run code tests. An
engineer must inspect it and test it before applying. This feature is local
only, never invoked automatically by GitHub Actions.

## Guarantees and limitations

- No Work quota reset, model identity spoofing, session-token rotation or rate-limit bypass.
- Only allowlisted jobs; no arbitrary shell commands from issues, prompts or AI.
- No secrets sent to AI; no unreviewed cross-repository commits or deployments.
- Report artifacts are time-limited; GitHub-hosted runners are ephemeral. Longer-lived queues require a separate persistent worker (the Studio already has an independent SQLite production queue).
- An *audit* is not an APK, film or proof that a strategy matches books. No bot trading, publication or payment authorization.
- Public-repository standard GitHub-hosted runners are generally free within applicable GitHub terms and abuse/rate limits; this is not infinite compute.

## Next integration gates

1. Approve separate CI presets for target repositories after verifying their existing workflow contracts.
2. Add a **user-controlled local worker** with a persistent volume and an explicit model provider.
3. Add proposed-code patches into a separate branch/PR *only after* human review, tests and bounded retries.
4. For Studio, reuse `studio.production_queue` and approved episode assets. Never substitute synthetic placeholders for real scenes or publish unreviewed content.
