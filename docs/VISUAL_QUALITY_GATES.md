# TWO WORLDS — Visual Production Quality Gate

### Engineering preflight ≠ professional illustration

The initial renderer uses original deterministic placeholder drawings.
This new premium pipeline accepts real painted or generated assets, layered into
depth planes, and allows a camera move and atmosphere per shot. Reaching
professional quality requires professionally designed *source illustrations*.

**Mandatory review for the first public episode**
- [ ] Work has locked the story/script; no autonomous rewrite by this studio.
- [ ] All bodies/occupants/languages match canon.
- [ ] Stable reference sheets exist for all recurring character bodies.
- [ ] Wardrobe, scars, anatomy and expressions consistent in every panel.
- [ ] No watermarks, signatures, copied IP, or unlicensed reference images.
- [ ] Background at least 1080×1920 (ideally overscan 1536×2730).
- [ ] Foreground PNGs have valid transparency; no visible halos or cutout seams.
- [ ] Each layer marked approved **only after actual human review**.
- [ ] Rights documented and verifiable.
- [ ] Camera parallax does not expose transparent canvas edges or crop heads.
- [ ] Mobile titles/subtitles remain legible at actual phone size.
- [ ] Animation has deliberate visual action and shots are not reused misleadingly.
- [ ] No clipped dialogue, inappropriate audio levels or audio/video drift.
- [ ] 9:16 H.264/AAC, 1080×1920, 24fps and correct duration verified.
- [ ] AI-generated media label considered according to platform rules.
- [ ] Owner approves publish separately from technical/QC approval.

**Automation safety:**
\`studio.production.scheduled_episodes\` now refuses scheduled premium
production without an accepted layered visual package for EVERY scene,
including documented rights and appropriate image resolution. Source images
may still be reviewed and video previews made with technical placeholders,
but those don't satisfy the schedule gate.

The GitHub Actions workflow tests and produces a PROCEDURAL LAYERED DEMO
to prove rendering works; the demonstration is **not** a polished comic episode
and is never marked publishable.

## Cost and scalability

FFmpeg/Pillow/NumPy layer compositing is free software and runs on CPU.
Professional consistent new illustration requires a supported GPU image
generator (e.g. the optional local ComfyUI backend) or real human-made art.
GPU access, checkpoints, licenses and storage can have costs. GitHub-hosted
CPU Actions runners do not provide guaranteed free GPU image generation.

To avoid bloating Git:
- Store scripts/contracts/metadata and small tests in repo.
- Keep large layered art, voice and completed MP4 in approved external
  storage or expiring CI artifacts; store checksums to detect changes.
- Never commit credentials or unlicensed third-party assets.
