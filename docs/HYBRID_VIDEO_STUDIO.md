# TWO WORLDS — offline hybrid video mastering (CPU)

This feature adds a reproducible **actual video export** route for the six STORY GLITCH stories. It can splice licensed pre-generated image-to-video clips with real still artwork (animated as slow Ken Burns camera movement), align recorded voice-over to individual shots, add optional music, and export H.264/AAC + SRT + QA receipt. **It does not generate images or AI video, acquire GPU access, make scripts, approve media or post videos.** Existing `studio.production` gates remain unchanged.

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

### Production guidance

- Start with a 9:16 still per shot. Use `push_in`, `pull_out`, `pan_left`, `pan_right` or `static` for CPU-based camera motion.
- Generate 3–5-second action clips separately **only when a real supported GPU or approved external provider is available**; add with `clip`. Voice and music are optional; a silent AAC track is added to preserve platform compatibility when no audio supplied.
- All clips are scaled/cropped to the requested 9:16 resolution, normalized to constant FPS and padded by holding the final frame if necessary. No interpolation or fabricated actor movement is claimed. Shot cuts are direct, not dissolves.
- Narration must fit within shot duration: the preflight rejects overlong speech rather than silently truncating it. Soundtrack and voice use an audio limiter to reduce clipping; this is not a substitute for listening and loudness mastering.
- Inspect the exported MP4/SRT/JSON on a real phone before publication, verify voices/rights and manually approve. The JSON receipt carries source and video hashes, duration, media mix and an explicit `published: false` field.

### Compute constraints

Standard GitHub-hosted CI runs CPU-only. Kaggle may provide T4/P100 GPU access under availability/quota restrictions, but no guaranteed production-grade image-to-video model fits unmodified on one 16 GiB T4. The official Wan2.2 TI2V-5B recipe requires **at least 24 GB VRAM** and is not silently scheduled as a free Kaggle job. This is deliberately a dependable *assembly/QA pipeline*, not a promise of free unlimited GPU inference.

## Smooth-camera correction (2026-11-04)

Feedback on the original 9-second pilot revealed visible jitter from aggressively stepped FFmpeg `zoompan` crop positions. **Still-image shots now use `studio.smooth_camera`:**

- Subpixel floating-point affine positioning, independently sampled from one source image with Pillow bicubic interpolation. No integer crop stepping or accumulation of resampling errors.
- Sinusoidal ease-in/ease-out, zero velocity at shot edges, restrained ~2.4% default zoom across an entire shot instead of the previous 10.5%.
- `camera_strength` per image shot, number **0..1**, default 0.45 (use ~0.25–0.4 for gentle human-interest drama). Use `camera: static` or strength 0 for absolutely steady image.
- Choose `fps: 30` for smoother mobile playback while retaining 9:16 H.264/AAC. At 24 fps motion is still subpixel smooth; 30 fps helps perception.
- This fixes camera stepping on *stills*; it does not stabilize shakes already baked into imported motion clips. Those need separate motion analysis and stabilization.

Existing master review/rights gates remain unchanged; no premium credits or GPU inference involved.
