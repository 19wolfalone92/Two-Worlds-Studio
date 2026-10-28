# TWO WORLDS — Character Animation Engine v1

## Status and purpose

**This is an actual deterministic 2D articulated-sprite renderer**.
It can rotate a head, move arms/forearms with inherited parent transforms,
translate the body, animate legs/cloth and render a grounded shadow, while
compositing in the existing CinematicPanel.

This is **not** automatic rigging of a single illustration, full facial
animation, IK-driven walking, mesh deformation or physically based lighting.
Work v1.1 has isolated *whole-actor PNG cutouts*, not artist-separated limbs.
Real cinematic motion requires preparing individual rig parts and a reviewer.

## Authorship boundary

Work: approved story/scene scripts and character reference paintings ONLY.
Studio: all art processing/generation, character rigging, animation, cameras,
scene compositing, voices, audio mixing, export and QA.

The 12 original scenes of Work's pilot remain intact. The studio uses the
private handoff as reference; no unreleased image or dialog is committed to
the public repository.

## Art prep: what the motion renderer requires

Prepare a **single canvas** shared by the entire rig (e.g. 1080×1920).
Split the character into separate **transparent** assets positioned in the
same pixel coordinates as the rest pose: torso, head (closed seams behind
collar), upper/lower arms, hands, legs, hair and optional cloth overlays.
The arm sprite stays at its resting shoulder on its full-size RGBA canvas.
It is not independently cropped and recentered. Do not change facial identity
when splitting the sprite. The renderer can crop internal non-transparent
bounds **for performance** without changing their coordinate system.

Each part has a normalized `pivot: [x,y]` (0..1). A parent's transform
carries all children: moving the shoulder carries the forearm and hand.
Higher `z` draws on top of lower `z`. The top-level track `__root__`
moves the whole actor. Rotations are degrees, clockwise in screen coordinates.
Keyframes use normalized scene time `at` from 0 to 1. Positions are delta
translations **relative to shared canvas width/height**.

Example (illustrative; all paths must be real reviewed PNGs):

```json
{
  "role": "character", "path": "E01/S05/actor_rest_rgba.png",
  "body_id": "earth_mira", "identity_version": "MIRA_EARTH_B01_v001",
  "rig": {
    "schema": 1,
    "root_pivot": [0.5, 0.78],
    "parts": [
      {"name":"torso", "path":"E01/S05/torso.png",
       "pivot":[0.5,0.55], "z":1},
      {"name":"head", "path":"E01/S05/head.png",
       "parent":"torso", "pivot":[0.5,0.38], "z":3},
      {"name":"upper_arm", "path":"E01/S05/upper_arm.png",
       "parent":"torso", "pivot":[0.58,0.52], "z":2},
      {"name":"forearm", "path":"E01/S05/forearm.png",
       "parent":"upper_arm", "pivot":[0.65,0.67], "z":4}
    ],
    "tracks": {
      "__root__": [{"at":0,"x":-0.03},{"at":1,"x":0.01}],
      "head": [{"at":0,"angle":-3},{"at":1,"angle":10}],
      "upper_arm": [{"at":0,"angle":0},{"at":1,"angle":-42}],
      "forearm": [{"at":0,"angle":0},{"at":1,"angle":-18}]
    },
    "ground_shadow": {"center":[0.5,0.87],"radii":[0.14,0.025],
                      "opacity":0.22,"blur":0.025},
    "environment_light": {"rgb":[85,112,160],"strength":0.12}
  }
}
```

In production, each rig part additionally requires explicit
`review_status:approved`, rights/rights_note and `sha256`; the master
remains subject to the Work script fingerprint and approved voice recording.

### Keyframe / schema limits

* Between 2 and 18 RGBA sprites; all match the base actor canvas dimensions.
* Parent names must exist and be acyclic. Sprite file paths must resolve
  under the approved asset root, and SHA-256 must match when supplied.
* 1–80 increasing timestamps per track, each in [0,1].
* Rotation is limited to ±150°, scale to [0.65,1.50], translation
  to ±60% of canvas. These are safety bounds, NOT good-acting defaults.
* Per-frame motion uses smoothstep interpolation (deterministic).
* A blurred contact-shadow ellipse is optional and must be positioned by an
  artist on the ground plane. Environment light is an optional subtle RGB
  tint; true scene relighting still requires artist input.
* Existing legacy `motion` and new `rig` are mutually exclusive to avoid
  double-applying animation.
* Invalid/missing art does not silently fall back to a fake good-looking shot.

## Missing for cinematic acceptance

1. **Actual separated body parts:** Work v1.1 gives 14 whole-character
   layers, not joint sprites. Someone must make high-quality layer separations
   without changing character likeness.
2. **Interaction:** mirror glass reflection/occlusion and hand contact need
   authored companion layers, coherent depth ordering, and keyframed timing.
3. **Face:** existing legacy blinking overlay is disabled if an aligned
   eye-erasure mask is absent. Eye/mouth animations should be artist-drawn.
4. **Quality:** compare generated motion with real character and location
   references, and have a human reviewer approve the look and movement.

## Test strategy

Unit tests cover hierarchy, deterministic frames, validation failures,
per-frame changes, master approval, and a full CinematicPanel render using
synthetic RGBA sprite fixtures. The tests are **not** proof that the real
Work pilot has been reanimated or is ready for TikTok.

Before scaling to 36 episodes, replace an 8–12 second excerpt using
owner-reviewed rig artwork, demonstrate a moving head, arm and grounded
environment, verify face continuity and submit the actual MP4 for approval.
