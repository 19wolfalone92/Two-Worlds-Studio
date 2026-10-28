# TWO WORLDS — Creative AI Studio: user-owned local engines

## What is installed

- **Art director:** `studio/art_director.py` reads `story/visual_bible.json` and
  scene body identity, generates deterministic scene prompts and records reference SHA-256.
  No automatic visual quality or facial match certification.
- **Optional illustrator:** ComfyUI API backend with a default SDXL text-to-image
  workflow in `templates/comfyui_sdxl_api.json`, only when `--backend comfyui`.
  The server and **licensed checkpoint** are not included. To get consistent
  characters, create approved multiview reference art and integrate a compatible
  IP-Adapter/ControlNet reference-conditioning workflow; the stock workflow does
  **not** include reference nodes. Multi-person reference scenes require
  bespoke workflow and manual compositing.
- **Story writer:** `studio/story_writer.py` uses locked narrative beats from
  `story/season1_beats.json` and a read-only story snapshot at
  `story/story_memory.json`. The default `outline` engine builds
  deterministic episode files from previously written story treatments.
  `ollama` engine (optional local model) can rewrite the English dialogue,
  headings and captions without permission to change body assignments or actions.
- **Safety:** generated scripts are written to `drafts/`, excluded from git,
  and tagged `editorial_status: draft_unapproved`. Release and publishing
  approvals are separate and unchanged.

## Run locally (free software, hardware needed for model inference)

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m studio.story_writer --episode E03 --engine outline
python -m studio.art_director --episode episodes/E01_THE_REFLECTION.json
```

To enable actual AI dialogue drafts, start Ollama with a local instruct model
on hardware you control:

```sh
ollama pull qwen2.5:7b
python -m studio.story_writer --episode E03 --engine ollama --model qwen2.5:7b
```

Default endpoint: `http://127.0.0.1:11434`. Any external URL
transmits the script/canon to that server. Do not trust unknown endpoints.

To enable actual AI image generation, start ComfyUI with a suitable
SDXL-capable checkpoint and sufficient GPU memory:

```sh
export COMFYUI_URL=http://127.0.0.1:8188
export COMFYUI_CHECKPOINT=your-licensed-sdxl-model.safetensors
python -m studio.art_director --episode episodes/E01_THE_REFLECTION.json \
  --backend comfyui --width 768 --height 1344
```

Prompts and reference images are saved in `dist/art/E01`. Generated
images must be visually approved, rights-cleared and staged under
`assets/` with a `background` path in an approved episode, before
the offline motion renderer can use them in final composites.

## Critical limitations

- GitHub's hosted Actions CPU runners do **not** grant a free continuous GPU
  for image generation or a local Ollama server; no automatic AI inference runs
  in CI. CI exercises deterministic outline and prompt preparation only.
- These systems do not guarantee a stable face solely from seed/description,
  or perfectly plot-hole-free scripts from an LLM. Human review remains required.
- Story treatments are locked per episode. The generator is not yet a self-directed
  multi-season novelist; later expansion needs episode planning, facts ledger,
  semantic continuity checking, creative QA, persistent model hosting and quotas.
- No password or credential in source code. No automatic uploads/publication.
