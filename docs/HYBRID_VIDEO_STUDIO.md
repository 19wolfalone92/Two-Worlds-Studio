# TWO WORLDS — offline hybrid video mastering (CPU)

This feature adds a reproducible **actual video export** route for the six STORY GLITCH stories. It can splice licensed pre-generated image-to-video clips with real still artwork, align recorded voice-over to individual shots, add optional music, and export H.264/AAC + SRT + QA receipt. The CPU compositor can also consume locally generated character takes.

## Run

Requires Python 3.11+, FFmpeg + ffprobe, and Pillow (`requirements.txt`). Save all source media beside or below a manifest JSON, with relative paths only.

```bash
python -m studio.hybrid_master --manifest input/my_story.json --output dist/my_story.mp4
python -m unittest discover -s tests -p 'test_hybrid_master.py' -v
```

Example manifest (the files need to exist):

```json
{
  "schema": 1,
  "width": 1080,
  "height": 1920,
  "fps": 30,
  "editorial_approved": false,
  "rights_approved": false,
  "music": "assets/background_music.wav",
  "music_volume": 0.12,
  "shots": [
    {"image": "assets/bridge.png", "duration": 5, "camera": "push_in", "camera_strength": 0.35, "caption": "Той ночью мост был пуст...", "voice": "assets/line01.wav"},
    {"clip": "assets/bridge_motion.mp4", "duration": 4, "caption": "Но кто-то оставил шарф."},
    {"image": "assets/scarf.png", "duration": 4, "camera": "pan_left", "caption": "И она узнала его."}
  ]
}
```

Default mode exports a visibly identified **unreviewed preview**. For `--master`, `editorial_approved` and `rights_approved` must be true at the top level, each shot must have `approved:true`, speech assets need `voice_approved:true` and any music needs `music_approved:true`. Approval metadata must correspond to a real editorial/rights review; the tool never invents it. `--master` does *not* publish or certify professional quality.

### Phone delivery

Set `"delivery_profile": "mobile"` at the top of a hybrid manifest to select
576×1024 at 24 frames per second. The profile supplies the dimensions and frame
rate when those fields are omitted. Explicit settings must match the profile.
Other delivery sizes can still be specified directly.

Author frame-bound performances at the selected frame rate. The clip importer
normalizes full-length video takes to the delivery clock. The export receipt
records the profile, actual dimensions, frame rate and source hashes.

### Hand and object contact

The Blender renderer accepts `--contact-manifest /private/shot/contacts.json`.
Place an Empty on each contact surface in the trusted scene. Parent the actor's
marker to the corresponding hand or finger bone and the target marker to the
prop. Use a separate table marker for the resting phase.

```json
{
  "schema": 1,
  "contacts": [
    {
      "name": "held-object",
      "marker": "Finger pad",
      "target": "Prop handle",
      "start_frame": 25,
      "end_frame": 90,
      "max_gap_m": 0.004,
      "max_slip_m": 0.001
    }
  ]
}
```

Windows include both endpoints and must fit inside the requested render.
The check evaluates the animated markers on every frame, converts scene units
to metres, measures separation, and measures changes in the target's rotated
coordinate frame. A detached or sliding contact blocks export with its frame
number. The receipt records the measured peaks and the manifest hash. Marker
placement and tolerances remain part of scene authoring. Combine these checks
with the existing mesh collision checks to inspect surface intersections.
Save the Blender worker's JSON result beside the take and pass its relative
path as `clip_take.receipt` with `engine: blender_3d`. The compositor verifies
the take's native frame clock and contact evidence, includes the receipt in
the cache key, and carries those measurements into the final shot report.
For Blender and neural character takes with authored motion checks, unresolved
temporal or identity findings prevent `--master` export. A preview remains
available for inspection; passing contact geometry alone does not approve acting.

### Production guidance

- Start with a 9:16 still per shot. Use `push_in`, `pull_out`, `pan_left`, `pan_right` or `static` for CPU-based camera motion.
- Generate 3–5-second action clips separately **only when a real supported GPU or approved external provider is available**; add with `clip`. Voice and music are optional; a silent AAC track is added to preserve platform compatibility when no audio supplied.
- Clips are scaled and cropped to the requested 9:16 resolution and normalized to constant FPS. Only legacy clips without an action requirement can hold their final frame to cover a shortfall.
- Narration must fit within shot duration: the preflight rejects overlong speech rather than silently truncating it. Soundtrack and voice use an audio limiter to reduce clipping; this is not a substitute for listening and loudness mastering.
- Inspect the exported MP4/SRT/JSON on a real phone before publication, verify voices/rights and manually approve. The JSON receipt carries source and video hashes, duration, media mix and an explicit `published: false` field.

### Compute constraints

Standard GitHub-hosted CI runs CPU-only. Kaggle may provide T4/P100 GPU access under availability/quota restrictions, but no guaranteed production-grade image-to-video model fits unmodified on one 16 GiB T4. The official Wan2.2 TI2V-5B recipe requires **at least 24 GB VRAM** and is not silently scheduled as a free Kaggle job. This is deliberately a dependable *assembly/QA pipeline*, not a promise of free unlimited GPU inference.

## Local neural character takes

Choose `echo-flash` for a dialogue take from an opaque portrait and the locked speech WAV. The driver aligns the source audio to the shot with silence and generates a full portrait video. The compositor mixes the original WAV again at the locked offset. Check the Russian articulation, cheeks and jaw on the decoded output. Upstream reports 12 GB VRAM for Flash and up to 768×768 generation. [EchoMimicV3 runtime](https://github.com/antgroup/echomimic_v3) · [Flash checkpoint](https://huggingface.co/BadToBest/EchoMimicV3/tree/main/echomimicv3-flash-pro)

Choose `wan-animate` to transfer a filmed performance to a character. Supply an authorized MP4 that covers the shot, with the head, neck and shoulders visible, a locked camera and an initial pose close to the reference. The adapter uses the upstream face and pose preprocessing in animation mode. It expects compatible framing and proportions in the reference and driving footage. The 24 GiB of free VRAM and native BF16 checks are preflight floors. Actual memory requirements depend on the checkpoint, shot and runtime. [Wan2.2 Animate model](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) · [Preprocessing guide](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md)

Install the official runtime and checkpoints in an isolated local environment. For EchoMimic, arrange the base model, audio encoder and Flash transformer as in the upstream README. For Wan, include the native model directory and `process_checkpoint` assets. The main checkpoints are published under Apache 2.0. Review the terms of their dependencies and the rights to the performance inputs separately. A local run uses the operator's GPU and disk.

```bash
python -m studio.neural_character --model echo-flash \
  --reference /private/job/portrait.png --driver /private/job/line.wav \
  --duration 3 --voice-offset-ms 430 --prompt-file /private/job/direction.txt \
  --runtime /opt/echomimic_v3 --weights /models/echo-flash \
  --python /opt/echo-venv/bin/python --output /private/job/take \
  --execute-local-gpu
```

For `wan-animate`, point `--driver` at the actor MP4 and omit the voice offset and prompt. Use the official Wan checkout, its model directory and a Python interpreter from its environment. The command defaults to an offline plan without `--execute-local-gpu`. It writes the silent take and its receipt together in a new private directory. CPU tests exercise the adapter contract and actual MP4 import with fixtures. GPU inference and the acting quality of production footage require a separate run.

In the production manifest, put the result in `clip` and set `clip_take.engine` to `neural_character`. Use the matching `visual_profile`, the receipt's `clip_sha256` in `sha256`, and the relative `receipt`, `reference` and `driver` paths. Set the same shot duration used at generation. For EchoMimic, also set `voice`, `voice_offset` and `dialogue_text` to the locked WAV, offset in seconds and caption text. Then use the regular `studio.cli produce` route.

Author `qc_regions` in the final shot coordinates. Every neural take needs a static camera anchor and face and wardrobe references pinned to the input image. For dialogue, use an upper-face anchor and distinct `mouth`, `cheek` and `jaw` regions with the same speech windows. Place cheek and jaw regions outside the lips. For reactions, measure the face relative to the shoulders while checking the shoulders against the camera in the same windows. The decoded motion report lists time ranges that need review. A neural take with unresolved motion or identity checks can be reviewed as a preview and is blocked from `--master`.

## Smooth-camera correction (2026-11-04)

Feedback on the original 9-second pilot revealed visible jitter from aggressively stepped FFmpeg `zoompan` crop positions. **Still-image shots now use `studio.smooth_camera`:**

- Subpixel floating-point affine positioning, independently sampled from one source image with Pillow bicubic interpolation. No integer crop stepping or accumulation of resampling errors.
- Sinusoidal ease-in/ease-out, zero velocity at shot edges, restrained ~2.4% default zoom across an entire shot instead of the previous 10.5%.
- `camera_strength` per image shot, number **0..1**, default 0.45 (use ~0.25–0.4 for gentle human-interest drama). Use `camera: static` or strength 0 for absolutely steady image.
- Choose `fps: 30` for smoother mobile playback while retaining 9:16 H.264/AAC. At 24 fps motion is still subpixel smooth; 30 fps helps perception.
- This fixes camera stepping on *stills*; it does not stabilize shakes already baked into imported motion clips. Those need separate motion analysis and stabilization.

Existing master review/rights gates remain unchanged; no premium credits or GPU inference involved.

## Embedded Russian subtitles & ducked soundtrack (Studio v2)

Set `"burn_subtitles": true` in an otherwise valid hybrid manifest to
render TikTok-safe, large, outlined Unicode Cyrillic captions directly into
the H.264 frames. A separate SRT is **still** exported. The composer uses
local FFmpeg/libass and installed fonts; without libass the render reports
failure (never falsely claims embedded text).

Each shot optionally accepts exact measured timings instead of a full-shot
caption:

```json
{"subtitle_cues":[{"start":0.12,"end":1.21,"text":"Она вернулась."},{"start":1.31,"end":2.40,"text":"Но дверь была открыта."}]}
```

Cue start/end are seconds **relative to that shot**, must not overlap and
must fit the real shot clock. No sentence segmentation or guessed speech
timing is performed by the subtitle renderer. The ARCHIVE 17 triptych builder
requires burnt-in captions in all three episodes.

When both licensed music and real voice tracks are provided, speech-triggered
FFmpeg sidechain compression automatically ducks the music without changing
source WAVs. The receipt records `automatic_music_ducking` and
`captions_burned_in`; human review of tone, loudness and coverage is still
necessary. No external API or GPU involved.
