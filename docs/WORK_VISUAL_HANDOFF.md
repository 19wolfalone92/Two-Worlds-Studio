# TWO WORLDS — WORK → VISUAL STUDIO CONTRACT (v1)

## Ownership boundary (mandatory)

**ChatGPT Work owns the scripts, series story bible, episode outlines, dialogue,
character motives and canon.** This studio owns **only** visual production,
image consistency, art preflight, 2.5D animation, sound compositing and export.
Do not let scheduled GitHub Actions rewrite plot, create episodes, or choose
story events. A production job starts only with a Work-approved episode JSON.

The existing \`studio/story_writer.py\` is legacy/experimental and is **not**
invoked by this project's CI. Keep it disconnected from production unless the
owner explicitly changes this policy.

## Quality target

Visually original cinematic dark fantasy / modern urban thriller, not an imitation
of named movies, comic artists or game art. Rich painted textures, accurate
anatomy, consistent light direction, composition with focal point, depth,
shadowed architecture and compelling lighting. This is a **target**, never
a statement that the current programmatically drawn demos meet that bar.

- Deliver 1080×1920 progressive, 24fps H.264/AAC (30fps optional).
- Source art preferably 1536×2730 or larger, 9:16; separate RGBA layers
  for background, midground, character and foreground.
- Do not bake dialogue into image pixels. Studio draws separately.
- Avoid main facial details and critical story elements below ~85% height.
  Leave generous free area in center lower third for captions and
  far-right safe margin for platform UI. Review on a real phone.
- A scene without rights-cleared, reviewed, sufficiently large cinematic
  layers **fails the scheduled premium release gate**. It can render as
  a technical preview only.

## Required reference-sheet library per physical body

**Very important:** references follow the PHYSICAL BODY, not the transported
consciousness. The same body should always look like the same person.

For Mira's Earth body / Lyra's Noctaris body:
- Shared bone structure (similar visual ancestry), distinct scar / attire /
  training, stable eye shape, hairline, face proportions, nose, jaw and age.
- Character turnarounds: straight front, three-quarter, side profile,
  back, close face portrait; expressions (fear, grief, anger, joy,
  surprise); full body action poses, hands, costumes and details.
- A clear stable reference identifier and version: e.g.
  \`MIRA_EARTH_B01_v001\` and \`LYRA_NOCTARIS_B02_v001\`.
- Character style consistency is evaluated by an artist, NOT only seed/prompt
  and NOT by an unsupported claim of automatic facial verification.

For each extra supporting body repeat the same process.
Before any batch generation, approve the **character reference sheets**.
Use a local GPU ComfyUI reference-conditioned workflow or other licensed image
model with identity conditioning and appropriate hands/pose guidance.
No guarantee that every panel will pass a human likeness check.

## Asset contract

Store correctly owned/rights-cleared PNG/WebP/JPEG files underneath \`assets/\`.
Never hard-code absolute paths or download unknown images to run a build.

Work must attach one **cinematic** block to each scene it approves.
This extends, rather than replaces, existing required episode keys and continuity
metadata. Example (template only, NOT a new story or scripted episode):

\`\`\`json
{
  "id": "SCENE_ID_FROM_WORK",
  "duration": 4.0,
  "palette": "noctaris",
  "effect": "embers",
  "heading": "",
  "caption": "",
  "continuity": {
    "state": "post",
    "body": "noctaris_lyra",
    "occupant": "Mira"
  },
  "cinematic": {
    "layers": [
      {
        "role": "background",
        "path": "episode_id/scene_id/background.png",
        "depth": -0.85,
        "opacity": 1.0,
        "blend": "over",
        "sha256": "64-character lowercase sha256 computed on the actual file",
        "review_status": "approved",
        "rights": "original",
        "rights_note": "Original created for TWO WORLDS by ...; approved by owner"
      },
      {
        "role": "character",
        "path": "episode_id/scene_id/character_rgba.png",
        "depth": 0.35,
        "opacity": 1.0,
        "blend": "over",
        "sha256": "64-character lowercase sha256 computed on the actual file",
        "review_status": "approved",
        "rights": "owned",
        "rights_note": "Original owned character illustration, release version"
      },
      {
        "role": "foreground",
        "path": "episode_id/scene_id/foreground_rgba.png",
        "depth": 0.9,
        "opacity": 0.85,
        "blend": "screen",
        "sha256": "64-character lowercase sha256 computed on the actual file",
        "review_status": "approved",
        "rights": "commissioned",
        "rights_note": "Rights assigned under the artist commission agreement"
      }
    ],
    "camera": {
      "start": {"x": -0.15, "y": 0.0, "zoom": 1.05},
      "end": {"x": 0.20, "y": -0.08, "zoom": 1.22}
    },
    "vignette": 0.18,
    "grain": 0.06
  }
}
\`\`\`

The named persons in the JSON are **illustrative continuity fields**, not
instructions to write a new episode. Rights values supported: \`original\`,
\`owned\`, \`commissioned\`, \`licensed\`. The owner must actually have the
rights; a JSON label cannot grant copyright.

## Rendering and checks

Preflight technical inputs (not an artistic review):

\`\`\`sh
python -m studio.premium_assets --episode episodes/E01_THE_REFLECTION.json --report dist/visual_review.json
\`\`\`

When all scenes include approved 1080×1920 layered art:

\`\`\`sh
python -m studio.premium_assets --episode episodes/YOUR_WORK_APPROVED_EPISODE.json --publish-gate
python -m studio.production --episode episodes/YOUR_WORK_APPROVED_EPISODE.json --width 1080 --height 1920 --fps 24 --quality master --output-dir dist
python -m studio.review_board --episode episodes/YOUR_WORK_APPROVED_EPISODE.json --output dist/storyboard_contact.png
\`\`\`

The production command alone makes a preview; **only the scheduled release gate**
requires premium-approved art, full rights and explicit approval flags. No TikTok
publishing is implemented. Prefer human final approval before any posting.

## Art QC procedure

For each shot: compare with character sheets (eyes/nose/jaw/hairline/scar/
armor details); screen for broken fingers, extra limbs, wrong hand anatomy,
misplaced pupils, distorted weapons and architecture, duplicated eyes, text
artifacts, discontinuity of lighting and perspective, cutout seams and missing
foreground alpha. Verify visual storytelling works without sound, and after
adding captions. Check exact scene body/occupant against canon, then approve.

Automated \`studio.premium_assets\` checks hashes, dimensions, brightness,
contrast, alpha, codec and licensing metadata; it CANNOT certify anatomical
correctness, original ownership, expressive acting or artistic merit.
Its approval flags must be entered by a human who has actually reviewed assets.

## Hand-off to Work

Work should hand the studio:
1. LOCKED episode scene JSON (complete scenes, dialogue and continuity).
2. Approved physical-body character sheets, environments and designs.
3. Layered transparent PNG assets / background plates, per scene.
4. Art-rights provenance, each panel's version + hash, shot/camera direction.
5. English voice assets/music with usage rights, or clear guidance to defer audio.

The studio returns: QC report, review contact sheet, MP4, captions, soundtrack,
source manifest and **review-only** publishing package.

## Encoder profiles

`--quality preview` (default) uses quick CPU rendering and CRF 23.
`--quality master` uses x264 slow / CRF 18, H.264 4:2:0,
48 kHz stereo AAC 192 kb/s. **Master is blocked** without 1080×1920+
approved premium art and at least 24 fps; these are encoding targets,
NOT proof that illustration, sound or artistic direction is professional.
A human QC pass still precedes TikTok delivery.

Large high-quality renders are CPU expensive and should run on a dedicated
renderer instead of assuming free CI minutes or a reliable cloud GPU.

## Additional master-release contract

Before the art enters production, Work returns a locked episode. Generate
the hash using `python -m studio.work_contract --episode episodes/FILE.json`.
Only the actual owner may record genuine approval:

```json
"work_lock": {
  "origin": "chatgpt_work",
  "script_sha256": "sha256 reported by the lock command",
  "editorial_approved": true,
  "approved_by": "actual owner reviewer"
}
```

For every on-screen recurring physical body, provide `body_id` and
`identity_version` in the `character` cinematic layer, and maintain
`story/identity_registry.json` with **three approved rights-cleared angles**
(front, three-quarter, profile), including matching SHA-256. This system
validates **which** identity is referenced, not whether faces visually match.
Artist/owner manual identity review remains required.

Master audio uses `audio/voice_registry.json`: speaker IDs follow acting
**consciousness** (Mira speaks with Mira's profile while inhabiting Lyra's
body). Provide each spoken scene's `voice` metadata including approved
recording path (relative to `assets`), `speaker`, `voice_version`,
`sha256`, rights and provenance. Clips must fit within scene durations;
master audio automatically mixes approved voice/music into an AAC stereo
track. Synthetic `espeak` is for previews only, never master.
Existing preview rendering needs none of this extra data.

Video output also gets quantitative dark-frame, white-frame and lack-of-motion
diagnostics. These flag possible production issues; not a quality guarantee.

## Automated visual asset staging

Work provides a locked episode and a per-panel **image map**. This is only
an artwork handoff map; the studio **must not** invent dialogue or scenes.
Sample structure:

```json
{
  "schema": 1,
  "episode": "E77",
  "script_sha256": "script SHA from studio.work_contract",
  "panels": {
    "S01": [
      {"role":"background","source":"S01/background.png"},
      {"role":"character","source":"S01/character_rgba.png","body_id":"earth_mira"},
      {"role":"foreground","source":"S01/foreground_rgba.png"}
    ]
  }
}
```

Complete **every actual Work-authored scene** in the map, using only its IDs.
`studio.asset_ingest` copies artwork into immutable fresh asset paths,
adds camera/light recommendations but never touches story, continuity, timing,
dialogue or captions. Its SHA-check confirms the Work script survived staging.
All new layers are **unreviewed** by design.

```bash
python -m studio.asset_ingest \
  --episode episodes/WORK_LOCKED_EPISODE.json \
  --image-map input/image_map.json --source-dir input/artwork \
  --assets assets --output dist/staged_episode.json
python -m studio.production --episode-file dist/staged_episode.json \
  --width 540 --height 960 --fps 12
```

For premium character motion, each transparent character layer may contain
`motion.breath_scale`, `motion.sway_x`, `motion.sway_y` and `motion.cycles`.
Optional `motion.blink` is a separately designed transparent, rights-cleared
eye overlay with normalized blink times, and `motion.mouth` is an approved
open-mouth overlay. The corresponding scene must have a **recorded voice asset**
for mouth cue timing. Lip animation uses energy-based open/closed cues, not
realistic phoneme-accurate performance. Human visual approval is mandatory.
