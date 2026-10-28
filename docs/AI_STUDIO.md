# TWO WORLDS Studio — Visual Models, Not a Script Writer

ChatGPT Work exclusively authors new episodes, dialogue and series story logic.
The now-removed experimental local Ollama story-writing code is **not part of
the production system**. The studio consumes Work-approved scene data.

## Professional art architecture
- `studio/art_director.py` generates versioned identity-consistent **prompt packs**.
  It binds art to the physical body, not to an exchanged consciousness.
- `story/visual_bible.json`: stable descriptions and visual style; no reference
  sheets currently exist in the repository.
- `story/identity_registry.json`: stable physical-body identity versions and
  front/three-quarter/profile reference images, hash-pinned and human-approved.
- **ComfyUI** generation is optional. You must supply a licensed checkpoint,
  compatible local GPU/server and image reference conditioning custom workflow
  (IP-Adapter or equivalent). **The default text-to-image workflow does not
  actually enforce matching facial identity.**
- Generated artwork is **unreviewed** by default. It cannot be master-published
  until the owner confirms rights, likeness consistency, facial anatomy,
  color/light continuity, and each image is stored in `assets/`.

## Pipeline
`Work locked episode → art shot planner → illustrated reference approval →
background/character/foreground assets → 2.5D render → licensed actors/audio →
MP4 QA → owner review → publication handoff.`

No GPU is granted by GitHub Actions. Hosted inference, disk space, model weight
downloads, and model licensing may incur costs, even for free software.

## Examples
```bash
python -m studio.visual_director --episode episodes/E01_THE_REFLECTION.json --output dist/shot_plan.json
python -m studio.art_director --episode episodes/E01_THE_REFLECTION.json --output-dir dist/art
# Optional only on YOUR server running ComfyUI and a licensed checkpoint:
python -m studio.art_director --episode episodes/E01_THE_REFLECTION.json --backend comfyui --checkpoint my-licensed.safetensors
```

The actual `--quality master` export is guarded by Work editorial lock,
license/identity provenance, reference views and genuine recorded approved
audio. No API keys or publishing actions are stored in this repository.
