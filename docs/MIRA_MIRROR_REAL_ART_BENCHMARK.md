# Real Mira / Lyra mirror acting benchmark (PRIVATE Work v1.1 assets)

**Status: prototype for artistic review — not final EP01.**  
The existing Work pilot was technically valid but animation/artistic motion was
not acceptable. Work is creative-only. The studio takes existing Work 1.1
artwork, joints it, and renders a 10-second sequence without new story or faces.

## Verified actual source

The private `TWO_WORLDS_Studio_Handoff_v1_1.zip` from Work is **not**
committed to GitHub because the repository is public. The renderer reads its
unpublished original images directly from ZIP. The art includes Mira on Earth,
Lyra beyond the glass, the original mirror room and physical contact-hand art.

The resulting shot is an experimental montage of pre-existing **S03 + S08**
visual actions; it does not edit/reorder the published canon or replace Work's
episode script.

## Run locally/on a private worker

```bash
pip install -r requirements.txt
pip install -r requirements-cinematic.txt
python -m studio.private_mirror_parallel \
  --handoff /PRIVATE/TWO_WORLDS_Studio_Handoff_v1_1.zip \
  --output /PRIVATE/TWO_WORLDS_MIRA_MIRROR_10s_VISUAL_TEST.mp4 \
  --fps 24 --seconds 10 --workers 4 --scale 1.0
```

Requirements: Python 3.11+, FFmpeg, Pillow, NumPy, optional OpenCV installed
from `requirements-cinematic.txt`; enough RAM for 4 × 1080×1920 compositors.
The parallel renderer runs on Linux using `fork` to share reference art pages
between workers. For constrained runners, set `--workers 2 --scale .6`; the
final export is still 1080×1920, but detail is interpolated from a lower
working resolution and **must not be called a native UHD source**.

## What it actually animates

- Two hand/forearm polygons manually separated from genuine Work RGBA art;
  original face pixels are preserved (no cloned or synthesized faces).
- Slow approach by Mira to glass and independent right-arm motion for Lyra.
- Secondary matching of dim Earth-side light, subtle glass highlights and
  environmental occlusion via original mirror frame.
- Cinematically cut contact shot with both Work original opposing palms;
  both hands approach one stationary glass plane without passing through it.
- Original Work Mira S03 voice at 2.15s; rendered brown-noise ambience.
- **No eyes are animated** until a correct anatomically matched eyelid rig is
  supplied: this deliberately avoids the deformed blinking in pilot v1.1.
- Every frame is composited from actual Work images, not anonymous generated
  placeholder characters.

## Limits and release policy

This is **not yet** full character acting or a complete new episode: there is
no learned skeleton, mesh-skinning, inverse kinematics walking, facial
performance capture, facial phoneme rig, motion generation, camera calibration
or verified realistic reflections. It uses handcrafted 2D rigid sprites and
artist-tuned compositions. Work's original character art and hand plates are
excellent; the new technical clip shows only an improved starting point.

Run includes a final full FFmpeg decode and writes a JSON report with
sha256 references and `owner_approved:false`. A green automated test does
not certify continuity of facial identity nor fine aesthetic quality.

**Neither GitHub workflow nor this renderer publishes to TikTok.**
An actual owner-reviewed shot and explicit publication consent are mandatory.
Do not approve additional episodes or release simply because checks pass.
