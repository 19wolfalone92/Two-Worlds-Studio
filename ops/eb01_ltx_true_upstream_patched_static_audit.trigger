Compare every real memory and stage telemetry patch against official upstream SHA 50f0aef1297574a4b29acbdb874e3a363a6d5f2e, py_compile fully patched code and 8-step config, no GPU, no image or voices. Fail closed before brother GPU use.

2026-11-14 audited revised bounded process monitor, test assertions synchronized with authoritative model code, now verify real official upstream patches and Python compilation.

2026-11-15: real T4 fp16 VAE produced 33 frames but visual RGB gate flagged all. New controlled scientific precision A/B: VAE encoder+decoder FP32 sequential offload, 2B transformer and T5 FP16; static audit tool d9dcfc19 must patch EXACT SHA-pinned LTX source and compile. No actual GPU, no fake video.

2026-11-15 controlled real I2V motion test: original photo/prompt/seed and 33-frame distilled 8 steps identical, image_cond_noise_scale changed 0.025 -> official default 0.15, explicit conditioning_strengths=[1.0], FP32 VAE, measure real motion and RGB QA. Latest verified V8 30-source private mux CPU 17 tests successful GH 43718169749. Full static audit, do not submit GPU until source compile/tests pass.
