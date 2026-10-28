# Mira / Lyra mirror shot — REJECTED animation and corrected staging

## Owner review verdict (29 Oct 2026)

**The original 10-second preview from PR #12 is REJECTED as an animation master.**
Mira is staged from the WRONG angle. In the composition the viewer must see
**Mira's BACK** as she faces the mirror; **Lyra remains on the OTHER side of
the glass** and faces Mira. This is one coherent geography, not two front
portraits sliding over an illustration. No final pilot/episode should use
that obsolete preview.

The existing Work v1.1 image `mira_dialogue_rgba.png` shows the *old* actor
angle. Cropping or horizontally flipping it cannot produce a true anatomically
correct rear view. The new path therefore **refuses to render** without
additional *private* authored rear-view art. This is a quality safeguard,
not a claim that new rear-view character art already exists.

## Required private rear-view art (not in public GitHub)

Prepare the SAME Mira (same hair silhouette, clothes, posture, body proportions)
seen from behind, standing on the EARTH side of the mirror, with the following
separately authored RGBA full-canvas layers:

- `torso`, `head`, `upper_arm`, `forearm`, `hand`,
  `left_leg`, `right_leg` (minimum seven; 8–12 recommended).
- Transparent shoulder/elbow/neck/hip seams, clean joint paint beneath layers.
- Authentic rear-view garment fabric and hair volume, rather than a mirrored
  frontal face or two large rotating cutout polygons.
- Consistent scene-scale floor contact and shadow; no floating feet.

All layers share one 1080×1920 canvas with the same actor coordinates.
Create a private JSON manifest in the existing `studio.character_engine`
`schema=1` format. It must include `body_id: "earth_mira"`,
`camera_view: "back"`, `facing: "mirror"`, and SHA-256 for **every**
sprite. The code also demands distinct changing motion tracks on
`__root__`, `head`, `upper_arm`, `forearm`, `left_leg`, and `right_leg`,
plus `ground_shadow`.

Metadata and SHA checks cannot prove that the artwork actually shows the
rear of the character: final visual owner review remains mandatory.
No Work art, faces or source voices are added to the public repository.

## Performance requirements: real character acting, not sliding stickers

- **0–2 s:** Mira stands 3/4 to camera BACK (face not exposed), looks toward
  the mirror. Her centre of gravity and weight shift are visibly grounded.
- **2–5.65 s:** She cautiously approaches glass with at least two believable
  foot contacts. Shoulder, elbow, wrist and head tracks are independent, but
  remain anatomically connected. Each planted foot holds position on the floor,
  without foot skating. Camera respects her rear silhouette.
- **Lyra:** Visible beyond the glass; she notices Mira and raises her hand.
  Layers remain behind the original mirror frame and in correct depth order.
- **5.65–6.2 s:** A motivated cinematic cut from back-view two-actor shot to
  contact shot, NOT a dissolve that morphs bodies through glass.
- **6.2–10 s:** The two hands approach one glass plane from opposite sides;
  no hands passing through, broken anatomy or implausible reflections.

The 2D engine supports authored pivots and keyframes; it does NOT synthesize
real anatomical motion from the original whole-character PNG. A properly
prepared rig and human animation polish are prerequisites for acceptable
movement. **Existing success on CI and a decodable MP4 are not artistic QA.**

## Private native-resolution command

```bash
python -m studio.private_mirror_parallel \
  --handoff /PRIVATE/TWO_WORLDS_Studio_Handoff_v1_1.zip \
  --mira-rear-rig /PRIVATE/mira_rear_approved_rig.json \
  --mira-rig-assets /PRIVATE/mira_rear_layers \
  --output /PRIVATE/TWO_WORLDS_MIRA_MIRROR_REAR_REVIEW.mp4 \
  --fps 24 --seconds 10 --workers 3 --scale 1
```

Python 3.11+, FFmpeg, `requirements.txt` and
`requirements-cinematic.txt` are needed. A missing rear rig, missing body
part, static required motion, wrong staging metadata, wrong sprite size or
mismatched SHA prevents output. Existing frontal Work art **never silently
replaces** an absent rear model.

## Acceptance gate

Review the FULL MP4 and frame breakdown at 0, 2, 4, 6, 8 and 9.5 seconds
at 1080×1920, 24 FPS. Explicitly confirm: (1) Mira back to camera and facing
the mirror throughout the wide shot; (2) consistent character design;
(3) weight transfer and non-sliding feet; (4) arm/wrist articulation and
no seam cracks; (5) Lyra on correct side of the glass; (6) clean glass
occlusion and continuity to the hands close-up; (7) no facial/eyelid warps.

The prototype has **not** passed that acceptance. No pilot replacement, EP02
production scale-up, main merge or TikTok posting without fresh owner approval.
