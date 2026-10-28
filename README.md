# TWO WORLDS — Motion Comic Studio

Original English-language dark-fantasy / sci-fi serial. **Independent of all Williams projects.**

## Status
**MVP / review stage.** A free, deterministic *motion-comic compositor* and pilot storyboard are included. It renders motion, depth, mirror glow, transitions, English caption overlays and a generated ambient audio bed. It does **not** generate new illustrated characters, realistic acting, voice dialogue, or upload to TikTok. Those are separate capabilities and are not represented as complete.

## Quick start
Requirements: Python 3.11+, FFmpeg on PATH.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python studio/render.py --episode episodes/E01_THE_REFLECTION.json --output dist/E01_preview.mp4 --width 540 --height 960 --fps 12
```

For full-size output change to `--width 1080 --height 1920 --fps 24`. Production-quality artwork should be placed in `assets/` and linked from the episode JSON using relative paths. Without artwork, the renderer displays **abstract procedural stand-in scenes** so the end-to-end pipeline can be checked without paid services.

## Production sequence
1. Finalize story bible and episode scripts; review continuity.
2. Prepare character sheets and illustration layers (rights-cleared).
3. Animate and render MP4.
4. Review video, captions, audio, continuity and license provenance.
5. After **human approval**, schedule via Metricool. Credentials must never be committed.

## Automation
GitHub Actions checks Python logic on pull requests and supports a **manual** preview render. It does **not** publish or generate fresh AI art on a schedule. A future automated scheduler requires dependable approved visual assets or a compliant image generator, persistent storage, a publishing integration and safeguards.

## Repository visibility
If this series and its scripts are not ready for public disclosure, make the GitHub repository **private** in Settings → General → Danger Zone → Change repository visibility.
