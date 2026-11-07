# WOLF AI 2.0 — один Android APK, без Termux

## Что в одном приложении

Native Android UI + offline SQLite history/tasks + Android Keystore GitHub token +
three allowlisted GitHub CI integrations + native llama.cpp (arm64) + the actual
Qwen2.5-Coder-0.5B-Instruct Q4_0 GGUF model **bundled inside the APK**.

The phone does NOT need:
- Termux, external local inference server or separate LLM APK;
- an external AI API, internet connection to chat, subscription or card;
- a computer to run offline chat.

It does need a recent 64-bit Android device and roughly **1 GB of available
storage** for the large APK and the first-run verified private model extraction.
Inference uses CPU, may be slower than cloud, and 0.5B parameters do not match
large proprietary models.

## Deterministic, reproducible supply chain

The CI job downloads exact upstream llama.cpp source commit
`e95426f16982054c379ca7bdc8186996dd22f0fe`, builds via the Android
NDK and links native `wolf_native` JNI library. Upstream license: MIT.

The official Apache-2.0 model is downloaded at CI build time from:
https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF

Exact GGUF:
`qwen2.5-coder-0.5b-instruct-q4_0.gguf`

SHA-256:
`9739055e046d62a937e5b7879012209ef40ebea8a1569a96028de491f3f091d5`

CI **fails closed** if its checksum is wrong, if the model is absent from
APK assets, or if `libwolf_native.so` is missing. Source and model licenses are
included in the APK assets.

After installation the app copies the GGUF from its own APK into private
application storage, verifies SHA-256 again and loads it with llama.cpp.
The user can chat on the phone without network once installed. App does not
execute terminal commands, invoke third-party runtime software or send
conversations to the network.

## GitHub

The user can choose to provide a *fine-grained*, short-lived GitHub PAT scoped
to **19wolfalone92/Two-Worlds-Studio** with Actions: read/write. The app then
runs only allowlisted workflows already present in the repository: Android APK
CI, Studio engineering preview and free-provider unit checks. **The user must
create and authorize this token**; this cannot be injected by ChatGPT or the
GitHub connector. The token is encrypted by Android Keystore, never sent to
the local LLM and never stored in the source repository.

## Limitations

No Work limit circumvention. GitHub Actions obey GitHub quotas.
No autonomous code-rewriting of the trading bot, no order execution, no video
publication, and no claims that the preliminary Studio engineering preview is
a finished movie. Compilation CI proves packaging/build, not interactive
performance on the user's Samsung device.

The APK is ARM64 only; no x86 emulator and no 32-bit Android.
This is a **large APK**. New debug releases may use a different signing key,
so automatic in-place updates of previous debug APKs may fail.

## Building locally

The supplied CI workflow is the reproducible build process. Locally, check out
exact llama.cpp revision under `mobile/wolf-ai/third_party/llama.cpp`, install
Android SDK 35, NDK 29.0.13113456 and CMake 3.31.6, and place the **verified**
official GGUF in `mobile/wolf-ai/app/src/main/assets/`. Then run:

`gradle --no-daemon -p mobile/wolf-ai :app:assembleDebug`

Do not commit the model binary or your GitHub PAT to the public repository.
