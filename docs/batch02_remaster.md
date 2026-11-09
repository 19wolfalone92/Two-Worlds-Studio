# Private Batch02 remaster production

The remaster reuses the studio's private Kaggle relay, cast references, FLUX
image worker, Qwen Russian narration, measured subtitle alignment, photographic
camera and short-form renderer. It has an isolated request directory and branch.
Earlier production requests, published posts and the existing schedule are untouched.

Object and hand-detail staging can now precede long cast instructions. This
prevents generic portrait language from crowding the requested story prop out
of the prompt. The option is explicit so earlier productions retain their prompts.

Static holds clone one bicubic composition rather than resampling it for every
frame. Approved scenes can be rendered into the verified cache while other shots
are being repaired. Cache-only operation cannot create or approve a final movie.

The narration clock tolerates ffprobe's microsecond rounding at exact video-frame
boundaries. Actual fractional frames still round up. Cached scenes must match
the current frame count within less than a frame. Completed FFmpeg artifacts are
flushed before probing or returning them.

The sampled audio module retains speech speed, edits only exterior silence and
places tracks on a 30 FPS clock. It uses the existing owned instrument bank and
harmonic palettes, with a quiet sidechained score. No paid API is introduced.

Validation: 21 targeted tests passed for camera stability and static composition,
frame-boundary rounding, sampled audio, prop-prompt priority, source isolation,
bounded repairs and sealed transport. Six real final films passed complete decode,
single-frame black detection, stream checks and source-exact subtitle comparison.
Measured final AAC loudness is close to -16 LUFS. Actual sources and final frames
were visually inspected; automatic recognition is not described as human listening.
Unpublished media, scripts, generated narration and detailed delivery reports stay
in the private handoff and are not included as public repository assets.
