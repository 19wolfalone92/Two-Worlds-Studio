# EB01 2026-11-12 — verified photo and motion recovery, NOT a final film

## Failure 1: photo worker upload
Old run #43160000340 stopped at Kaggle SaveKernel HTTP 400 while
trying to embed the existing full-size Kostya PNG in a doubly base64-encoded
worker source. API 400 alone did **not** disclose its internal validation
message; oversized request is the leading diagnosed cause, not proven exact
server response. No model inference occurred in that old failed run.

Root fix on `fix/eb01-photo-upload-small-payload-20261112`: worker receives
ONLY a SHA256 and tiny metadata for the original Kostya, never the binary
portrait. The original remains in private
`wolf19loner92/eb01-v6-canon-adad762a9834`; no public GitHub image upload.
CPU tests assert small source bytes, no Base64 PNG and reproducible checksums.

- New run #43175626373: **SUCCESS** (actual private T4 SDXL photo generation).
- Its image, although photoreal, was **visually rejected**: white/beige shirt,
  pale neutral background, too-tight headshot; violates approved black-lace
  gothic-office character design. Do not waste Wan GPU animating it.
- New targeted art-directed run #43176690484: **SUCCESS**. Its
  `RunDiffusion/Juggernaut-XL-v9` candidate now visibly depicts a real
  brunette in a black lace high-neck outfit, seated behind a walnut desk
  with dark vintage office decor and warm lighting. Some design details
  still require artistic review (hair is swept back rather than long/wavy,
  and character appears more mature). **No owner approval is claimed.**
- New Elizabeth ORIGINAL SHA256:
  `7c691765a9f7cf3c190d110f114ce7d3b8a2f39eb681ecfe3056817c8e9a58a1`.
  Original PNG remains in private Kaggle
  `wolf19loner92/eb01-v6-photoreal-e140674dabc5`.
- Original Kostya PNG unchanged SHA256:
  `447373a1ee3162dd81ab29e07f5cbff24dbe4bab44ddbbc7394213ecb6ca0afd`.
- Both GitHub artifacts contain **small contact/review thumbnails only**.

## Failure 2: true motion with neon corruption
- Genuine Wan2.2 49-frame 448x800 source passed MP4 decode in
  https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/43159416691.
- Visual examination found 3 conspicuous neon stripes at frames 46,47,48.
- A CPU FFmpeg salvage with only the last three frames trimmed preserves
  46 real AI-generated frames (1.9167 s at 24fps); no extra GPU spent.
  The trimmed clip is **only a technical motion pilot**, not finished E01.
- Regression gate in PR #69
  https://github.com/19wolfalone92/Two-Worlds-Studio/pull/69
  now detects unusual bottom-frame chromatic spikes and blocks any broken
  Wan clip before Studio V6 uses it for montage.

## Release policy
No full seven-scene episode until canonical characters and a convincing
real neural motion test pass VISUAL review; six synced spoken lines required.
Do not substitute a CI pass, collage or pan/zoom for full native animation.
Nothing from this recovery was posted on social media.
