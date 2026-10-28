# STUDIO FIRST — TWO WORLDS production change (October 2026)

## Ownership and objective

**ChatGPT Work delivers only creative canon, original story, final dialogue,
locked scene instructions, and initial character reference concepts.**
It does **not** render or publish future episodes.

**TWO WORLDS Studio is the one production engine for the entire show:**
receives the private Work handoff, reconciles the physical-body identity
registry, generates/edit illustrations on approved free GPU services,
prepares character-specific layers, handles real actor animation and
environment matching, composites sound and video, audits and stages MP4.

First pilot v1.1 is a technically valid **limited 2.5D** video, NOT the
professional visual-animation standard. We do not delete it: preserve its
12-scene Work handoff and cinematic source layers.

### Why the first pilot looks like sticker/slideshow animation
The actual Work v1.1 Studio handoff supplies 12 scenes with background and
character PNG cutouts, but no declared `layer.motion` rig or keyed acting
motion. Work rendered the video using its own external compositor. The main
Studio engine cannot reproduce that private renderer merely by processing
these 2.5D assets. Wide camera pans, breathing scales and parallax alone
will not make a character naturally walk, reach or interact.

### Quality 2.0: what changes
1. The existing Studio receives private Work handoff without committing the
   proprietary images into the currently PUBLIC GitHub repository. Use
   `python -m studio.handoff_audit --handoff path/to/Studio_Handoff_v1_1.zip`.
   This audits all referenced art layers, their hashes, supported roles,
   character-motion availability, foreground occlusion and grounding flags.
2. **No sticker-like double-eyes.** The old algorithm composited closed
   lids over still-open eyes. New renderer requires an aligned artist-painted
   `erase_mask` when using a blink overlay. An old, unmasked blink stays
   *disabled* in previews and blocks production masters.
3. **Character staging must be integrated into the location.** Art director
   must inspect contact shadow, spatial perspective and environmental
   light direction, with the results recorded in each scene's `animation`.
4. **Actual acting performance required.** Every character scene must use
   an authored rigged 2D performance, frame-by-frame animation, or approved
   external generated motion, with expressive cues and a real human review.
   Unmoving dramatic holds require an explicit review reason.
5. Create a single 8–12 second **cinematic benchmark scene** from existing
   Work v1.1 assets before doing full EP01 re-render or EP02 production.

### Performance metadata added for premium masters (example only)

```json
"animation": {
  "kind": "rigged_2d",
  "cues": ["gaze", "head_turn", "hand_action"],
  "integration": {
    "contact_shadow": true,
    "lighting": true,
    "perspective": true
  },
  "reviewed_by": "actual visual reviewer"
}
```

Fields describe verified work by an editor; setting them true does not
automatically make a scene attractive. Audit fails closed for missing
metadata but **does not constitute computer-vision proof of quality.**

### Masked blink format

```json
"motion": {
  "blink": {
    "path": "E01/S03/closed_eyes_rgba.png",
    "erase_mask": "E01/S03/eye_erasure_grayscale.png",
    "erase_mask_sha256": "<SHA-256>",
    "at": [0.45],
    "duration": 0.05,
    "review_status": "approved",
    "rights": "original",
    "rights_note": "Owner-reviewed original animated eyelid art"
  }
}
```

All eye sprites must be aligned to the **exact same pixel canvas** as the
character. A grayscale mask uses 255 to erase original open-eye pixels. The
closed-eye artwork is composited only after erasure; both then transform with
the actor sprite. No unmasked blink is shown in previews.

### What we still lack technically

No true articulated character rig, generated replacement motion frames,
depth-aware occlusion estimator or inverse-lighting relighter has yet passed
a real art-quality acceptance test. Those are substantial engineering/art
dependencies. A success report from unit tests does NOT mean cinema-quality
motion is already produced. The next production milestone is a 10-second
owner-reviewed quality test, not another 36 static episodes.

### Publication safety

First pilot is **not published** and remains unapproved.
Do not put Work's unpublished art or scripts into public GitHub.
Do not merge this proposal into main or trigger TikTok without owner review.
