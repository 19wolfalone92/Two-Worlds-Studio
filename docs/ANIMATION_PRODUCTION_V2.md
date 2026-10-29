# TWO WORLDS Studio — animation production v2 (review-only)

## Engineering changes (stacked on PR #15)

This branch adds FOUR **working** animation-related subsystems to the existing
Studio compositor. It does not replace the existing scripts, approved Work
continuity or private design gallery.

1. **Rear character drawing** — use the *existing* GPU factory's `back` job
   after the selected physical body has a generated front identity. The GPU
   job explicitly requests a TRUE 180-degree turn, with back of head and
   costume visible and NO face. The new code does **not** claim to have
   generated an additional rear image in this PR. Visually review the actual
   rear PNG and select the correct body before rigging.
2. **Automatic rig preparation** — `studio.animation_pipeline` segments a
   pre-matted upright silhouette into 14 individually hashed full-canvas
   RGBA layers, builds hierarchical pivots and writes `rig.json`. This is
   **automatic template segmentation**, not learned anatomical recognition.
   Seams, overlapping fabric and concealed joints need an artist's retouch.
3. **Body animation** — initial walk is baked from a screen-plane two-bone
   IK solver. During the foot's stance interval, its target is fixed in
   world coordinates; the opposite foot swings and lifts. Root, head, upper
   arm, forearm, thighs and shins animate independently. The existing
   CharacterRig renders actual per-frame moving layers. A second available
   mode, `DeformableActor`, bends a *continuous single-image pixel field*
   with authored controls instead of moving large detached body stickers.
   Neither approach proves realistic locomotion or a perfect non-skating
   foot on character art. Both require visual evaluation.
4. **Face and speech** — `PerformanceRig` transforms aligned artist-drawn
   eye/mouth overlays together with the head bone. Uses three eyelid states
   and time-coded viseme shapes when supplied. If only voice-energy is
   available, the fallback changes three mouth states and explicitly
   identifies itself as **not phoneme-perfect lip sync**. Eyes never blink
   on a back-of-head view. Closed eyelid plates must repaint/erase underlying
   open eyes; simple superposition is unsuitable.
5. **Cinematography** — existing depth camera/pan/zoom and parallax remain;
   optional `cinematic.lights` add up to four deterministic, keyframed
   2.5D colour/brightness passes. They are stylized screen-space lighting,
   not physically based 3D light transport.

## Private commands — actual material required

First produce/review the correct character's `back` view from the private
factory project, after its front view has been generated:

```bash
python -m studio.character_factory run-gpu --project /PRIVATE/NEW_DESIGN_v1 \
    --views back --max-images 1 --width 768 --height 1152
```

The available factory can carry out that GPU job on an appropriately
configured machine. This branch does NOT automatically dispatch GPU
workloads or spend API quota.

**Important:** the GPU gallery images were generated against a *matte studio
background*, not transparent. The CLI has an optional `--auto-matte` mode,
which can isolate a single full-body subject on a nearly uniform background
with OpenCV. It rejects complex backgrounds, partial figures and ambiguous
silhouettes; the result always needs hair/cloth-edge review. Without
`--auto-matte`, feed a professionally prepared transparent PNG. Do not
interpret a technically generated alpha channel as completed artist cleanup.

```bash
python -m studio.animation_pipeline \
    --image /PRIVATE/earth_mira_REAR_RGBA.png \
    --output /PRIVATE/mira_rear_rig_v001 \
    --body earth_mira --view back --facing mirror --auto-matte
```

Inspect `rig.json` and its 14 PNG parts. In this branch, the result is
**unreviewed** and explicitly disallows publication. After artist repair,
update part SHA pins and validate the rig with
`studio.character_engine.CharacterRig`; separate owner review is still
required before the mirror cinematography can be accepted.

The existing mirror-benchmark loader `studio.mirror_rear_rig` can consume
this rig's data if its private assets are on the same canvas as the shot.
Its old reference shot uses **1080×1920**; do not upscale a 768×1152 source
and call the resulting quality native.

## Optional continuous deformation (no detached part sprites)

A `character` layer can use `deform` *instead of* `rig` or `motion`.
Example, relative to the full-character canvas:

```json
{
  "role": "character", "path": "mira_rear_rgba.png",
  "depth": 0.4,
  "deform": {
    "schema": 1,
    "controls": [
      {"name":"head", "x":0.5, "y":0.22, "radius":0.2,
       "keys":[{"at":0,"dx":0,"dy":0},{"at":1,"dx":0.012,"dy":-0.01}]},
      {"name":"hip", "x":0.5, "y":0.63, "radius":0.26,
       "keys":[{"at":0,"dx":0,"dy":0},{"at":1,"dx":0,"dy":0}]}
    ]
  }
}
```

Requires OpenCV (`requirements-cinematic.txt`). Be careful with pose
amplitude; this is intended for small motions (breathing, cloth, posture),
not rotating a frontal picture into a rear view.

## Facial animation

An articulated `rig` character layer can include a `facial` dictionary:

```json
{
  "eyes": {
    "open":{"path":"eyes_open.png","sha256":"<actual 64-char SHA>"},
    "half":{"path":"eyes_half.png","sha256":"<actual 64-char SHA>"},
    "closed":{"path":"eyes_closed.png","sha256":"<actual 64-char SHA>"}
  },
  "mouth": {
    "rest":{"path":"mouth_rest.png","sha256":"<actual 64-char SHA>"},
    "small":{"path":"mouth_small.png","sha256":"<actual 64-char SHA>"},
    "wide":{"path":"mouth_wide.png","sha256":"<actual 64-char SHA>"}
  },
  "blinks":[1.25,3.4],
  "blink_seconds":0.18,
  "visemes":[{"at":0,"shape":"rest"},{"at":0.3,"shape":"wide"},
             {"at":0.45,"shape":"small"},{"at":0.6,"shape":"rest"}]
}
```

Every facial plate must be a full-canvas *RGBA* image with the same anatomy
and alignment as the character's head. Its visible region should fully
replace the corresponding anatomical patch, not simply draw outlines on top
of an already painted open eye. `visemes` are time-coded seconds, not the
normalized 0..1 animation position. No accurate phoneme alignment is
automatically generated by this branch.

## Cinematic lighting

At the `cinematic` scene level, specify screen-positioned lights:

```json
"lights": [
  {"x":0.49,"y":0.35,"radius":0.8,"rgb":[80,120,240],
   "from":0.05,"to":0.35,"flicker_hz":0.4}
]
```

The light's colour changes the screen-space composite in a deterministic
fashion. Existing camera position and depth still apply.

## Shot acceptance and unresolved blockers

**Correct staging:** Mira's back toward camera, Mira faces the mirror,
Lyra behind glass facing Mira. No camera-facing Mira or mirrored frontal
face. Stage their hands on opposite sides of the plane with clean occlusion.

Before advertising a finished 10-second clip: obtain private *true rear*
art; alpha-matte it; repair the auto-rig's joints (or produce independently
painted separated poses); preview body animation at real shot resolution;
check planted feet, eye replacement and hands; validate preserved identity
and continuity in the actual MP4; pass CI and have the owner approve.

Technical tests are not evidence of natural acting. **No new genuine rear
PNG, approved multi-part character or final MP4 is included here.** Nothing
is merged into main or posted to TikTok.
