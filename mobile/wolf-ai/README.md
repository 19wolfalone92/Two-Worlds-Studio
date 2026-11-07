# 🐺 WOLF AI Android v1 — free-first, native, offline-capable

**Working scope:** A real installable Android control panel for the WOLF AI checks
that already exist in the Two-Worlds-Studio repository. It has an **offline SQLite
task inbox** (up to 100 entries), deterministic Russian command classification,
GitHub Actions workflow dispatch, a latest-run status list, Android Keystore
encrypted personal token storage and an **optional localhost-only** llama.cpp
chat interface. No account registration beyond an existing GitHub account.

**Not yet supported:** full Work replacement, automatic code fixing and PR
creation, autonomous video generation, 24/7 agent hosting, trading,
TikTok publishing, offline model inference without a separately installed
local model, or bypassing ChatGPT subscription limits.

## Build Android APK without a PC

Open GitHub → [Actions: WOLF AI Android — Build APK](../.github/workflows/wolf-ai-android.yml).
Select **Run workflow** on main (once merged) or download the build artifact
from the pull request. Workflow compiles a native Android APK and verifies
package ID and permissions. Click the successful build → **Artifacts** →
\`WOLF-AI-Android-debug-...\`. Download the ZIP, extract the APK and install
on Android (allow installation of APKs for your file manager if prompted).

This is a *debug-signed APK*. Reinstalling a debug APK from builds that use
different ephemeral signing keys may fail with INSTALL_FAILED_UPDATE_INCOMPATIBLE.
Before uninstalling, note that v1 offline task data is local and not exported.

## Start working

1. Open **WOLF AI**. Without internet, you can add and delete tasks in the
   **Новые задания** area. They are local *plans*, not queued GitHub executions.
2. To see public CI results, tap **Обновить запуски**. No token required.
3. To start a CI check from the phone, generate a GitHub fine-grained PAT:
   GitHub → Settings → Developer settings → Personal access tokens →
   **Fine-grained tokens** → Generate new token.
   - Owner: your GitHub account.
   - Repository access: **Only selected repositories** →
     **19wolfalone92/Two-Worlds-Studio**.
   - Repository permissions: **Actions: Read and write**;
     Metadata: Read (automatically granted).
   - Give the token a short expiration and revoke it if the device is lost.
4. Paste the token only in the app's **Доступ к GitHub** field, tap **Сохранить**.
   This token is encrypted using a non-exportable Android Keystore key.
   It is not uploaded to the Studio repo and is never sent to a model.
5. Choose a project and \`audit\` or \`test\`; tap **Запустить проверку**.
   The app dispatches **only** the preexisting \`wolf-ai-audit.yml\` on main,
   then fetches run status using a **public GitHub read endpoint**.
   A successful dispatch is *not* proof the checks passed.
6. Tap a run entry to open GitHub's actual log and downloadable report.

## Scope of audits

- \`studio\`: preapproved syntax/readiness plus a selection of unit tests.
- \`shaira\`: existing npm check (including tests).
- \`williams\`: syntax only, **no bot code modifications and no trading**.
- \`williams-ai\`: syntax only, **no GPU/LLM training**.
- \`english-child\`: source inventory only, **no automatic APK build**.

The GitHub backend is finite and bounded. It does not accept arbitrary shell
commands, source modifications, repository tokens, or links from phone inputs.

## Optional local LLM (no cloud or paid APIs)

If you have a compatible GGUF model and \`llama-server\` installed separately
(e.g., in Termux), start the server bound to **127.0.0.1:8080**:

~~~bash
llama-server -m /path/to/your-small-quantized-model.gguf --host 127.0.0.1 --port 8080 -c 2048
~~~

The app checks \`http://127.0.0.1:8080/v1/models\` then sends the typed
question to \`/v1/chat/completions\`. It never calls a remote URL in this
section. The model file/server are *not bundled* with the APK.
Small locally supported 1–3B quantized models are more realistic starting
points for phones than large 7B+ models; performance depends on RAM/CPU.

Android network security only permits HTTP cleartext to localhost; GitHub
communication uses HTTPS. The Android app requests **INTERNET** only.

## Source and reproducibility

Android project: \`mobile/wolf-ai\` (Java, Android 26+, no third-party Android
runtime dependencies). CI: \`.github/workflows/wolf-ai-android.yml\`.
Existing Studio/Williams app/source code is not changed.

Free-first means **no required paid LLM/API/hosting**, not unlimited
GitHub Actions minutes, data transfer, GPU time, or model quality.
