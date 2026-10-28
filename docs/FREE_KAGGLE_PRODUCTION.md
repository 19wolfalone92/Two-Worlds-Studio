# Free GPU / Kokoro Setup — TWO WORLDS

## Current status (engineering truth)
The repository includes an executable **Kaggle GPU notebook**, FLUX.2 klein 4B
inference adapter, Kokoro TTS adapter and offline tests. **The notebook has not
yet been run on a real Kaggle GPU, and no professional reference portraits or
natural character voice recordings have yet passed human review.** We cannot
claim actual character consistency, production-grade pictures or a finished
episode from code alone.

The only user-required action to start the external free session is to open
a Kaggle notebook with an available accelerator. We do NOT need the user's
Kaggle password, API tokens or an expensive PC. We DO need actual cloud
inference, a real image review and approved Work script later.

### What is free, and what is not guaranteed
- Official FLUX.2 klein 4B open weights: **Apache-2.0**; source:
  https://huggingface.co/black-forest-labs/FLUX.2-klein-4B .
  The 9B Klein variant uses different license terms; avoid it in this kit.
- Official Kokoro-82M weights: **Apache-2.0**; source:
  https://huggingface.co/hexgrad/Kokoro-82M .
- Kaggle notebooks: free GPU access subject to availability, account
  eligibility, time limits, queueing, bandwidth, disk and policy restrictions.
  Official info: https://www.kaggle.com/docs/notebooks .
- GitHub Actions uses CPU for preview rendering; GPU inference and
  text-to-speech model loading are **not** run in the standard CI workflow.
- A 16GB Tesla T4/P100 lacks bfloat16, so the inference adapter uses
  float16 + CPU offload. This is an **unverified Kaggle compatibility path**,
  not a promise that the model will fit or run. Stop after OOM; try
  512x768 (or another allowed resolution), fewer concurrent workloads, or a
  separate free accelerator when available. Kaggle may refuse model
  downloads or run out of the ~20GB saved workspace.

## Owner procedure: from your phone

1. Open https://www.kaggle.com/code and create a notebook. Settings →
   Accelerator → available GPU, and enable internet if downloads need it.
2. Open the committed notebook
   \`notebooks/TWO_WORLDS_Kaggle_FLUX_Kokoro.ipynb\` from the repository
   (upload it as the Kaggle notebook source).
3. Run cells **1–2** with the opt-in switches set False. This clones the
   development branch, checks GPU access and dry-runs the job.
4. Set \`ENABLE_GPU_IMAGE = True\`. Run the image-generation cell for
   \`BODY="earth_mira"\`, \`VIEW="front"\`. One real image is generated.
   If the GPU is unavailable/insufficient, this stage stays incomplete.
5. **Visually inspect** the resulting PNG. Only if you approve the body
   appearance should it become the permanent front-face reference.
6. Use that image as \`REFERENCE\`; set \`VIEW\` to three-quarter and
   profile on subsequent runs. Save each, then visually check and select.
   This is image editing, NOT mathematical face-identity verification.
7. Set \`ENABLE_KOKORO_VOICE = True\`, run TTS cell. Listen and choose a
   stable synthetic voice preset for each fictional speaking consciousness.
8. Run packaging cell and download the ZIP from Kaggle Outputs.
9. In the Studio, import only rights-cleared, approved art and approved
   Work-authored screenplay. Use its \`asset_ingest\`, layered compositor,
   actor animation, soundtrack and QA before making a real master.

The notebook never publishes or sends outputs to TikTok, GitHub, Metricool
or third-party services except fetching declared public dependencies/models.

## Model commands (with GPU available)
\`\`\`bash
python -m studio.kaggle_flux --mode character --body earth_mira --view front --dry-run
python -m studio.kaggle_flux --mode character --body earth_mira --view front --width 576 --height 1024
# Optional consistent identity via EDITING a user-approved reference:
python -m studio.kaggle_flux --mode character --body earth_mira --view profile --reference /kaggle/working/approved_front.png

# Prepare a prompt pack from an existing Work-controlled episode:
python -m studio.art_director --episode episodes/E01_THE_REFLECTION.json --output-dir dist/art
python -m studio.kaggle_flux --mode scene --prompt-pack dist/art/E01/prompts.json --scene S01 --dry-run
\`\`\`

Note: \`episodes/E01_THE_REFLECTION.json\` is an old engineering fixture and
is NOT a Work-approved production screenplay.

## Kokoro voice synthesis

Preview sample:
\`\`\`bash
python -m studio.kokoro_tts --text "This is a voice test." --speaker Mira --dry-run
python -m studio.kokoro_tts --text "This is a voice test." --speaker Mira
\`\`\`

To voice a **Work-supplied** episode, Work can attach per-scene TTS metadata.
Example metadata, not a new scene or story:

\`\`\`json
"tts": {"speaker": "Mira", "text": "The Work-approved line exactly as written."}
\`\`\`

\`--episode\` then synthesizes only those explicit lines, leaves the original
scene and script unchanged, and returns unreviewed WAVs/manifest. Before any
master is rendered, an editor must verify the spoken lines **exactly match
Work's locked screenplay**, select the actual stable voice casting,
document rights to synthesized output and manually mark approved after
listening; \`audio/voice_registry.json\` defaults to unreviewed.

## Acceptance gate
- [ ] Kaggle model successfully generated at least one actual PNG.
- [ ] Human-approved front / 3/4 / profile references for the same character.
- [ ] At least 15 varied shots inspected for stable eyes, jawline, hair, scars,
  costume and pose. Fix errors manually; do not claim 100% likeness.
- [ ] Kokoro generates a WAV and a reviewer approves voice, pacing and accent.
- [ ] One **Work-approved** actual scene has full art layers, correct script
  lock, approved sound, a 1080x1920 MP4, and a clean visual/audio audit.
- [ ] Publication remains a separate owner decision.

### Privacy
The GitHub repository is currently public. Do not put unapproved scripts,
private images, passwords or unpublished proprietary source artwork in it.
Keep large Kaggle outputs in secured storage, upload only the required
rights-cleared assets when ready, and consider changing repo visibility to
Private before adding original intellectual property.
