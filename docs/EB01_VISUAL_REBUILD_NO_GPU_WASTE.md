# ELIZABETH BONES E01 — REBUILD / NO MORE WASTED GPU QUOTA

Status: **VISUALLY REJECTED, NOT FINAL**. The 23.208-second MP4 from Actions run #43079040704 passed technical container/audio checks but FAILED artistic quality. Do not publish.

## Actual observed defects, original MP4
- ~0.7s S00: a normal man at the desk rather than the cast's skeleton in the main pose.
- ~5.1–9.4s S02: Kostya becomes a grayscale human wearing white, not ivory skeleton in black.
- ~10.9s S03: Elizabeth switches to short black bob, different face and clothing.
- ~12.3s S04: a skeleton finally appears, but the suit has become white.
- ~15–19.6s S05: Elizabeth's face, hairstyle and costume drift again.
- ~21–22.5s S06: Kostya is a flesh-and-blood man, not a skeleton. BLOCK.

ROOT CAUSE: seven independent SDXL prompts and seeds, no shared identity
reference conditioning, no visual acceptance gate. FFprobe, six WAV files,
seven PNGs and valid AAC/H264 are technical checks, not a finished movie.
The current Studio preview uses smooth-camera stills, not real human/skeleton
movement. Calling it an animated film was incorrect.

## Correct production order
1. Lock two canonical subject images before scene generation: Elizabeth (same
   adult brunette, long dark wavy hair, natural skin, black lace blouse), and
   Kostya (full ivory skeleton with accurate skull, jaw and hand bones, BLACK
   tailored three-piece suit, white shirt, burgundy tie and black fedora).
   Keep source assets private, not in the public repository.
2. Create one shared master scene with both characters across the walnut desk,
   warm brass lamp and burgundy gothic office. Same cinematic realistic style.
3. Generate the 7 frames as identity-conditioned image-to-image variants of the
   same two subjects and room; NEVER 7 unconstrained text-to-image generations.
   S00 both; S01 Liz; S02 Kostya; S03 Liz; S04 Kostya; S05 Liz; S06 Kostya.
4. Deliver a simple contact sheet and human-review it. The CPU-only script
   tools/eb01_visual_gate.py binds approvals to individual asset SHA-256s.
   It rejects failed anatomy/face/clothing/composition checks and missing refs.
5. Render only ONE short (2–4 seconds) real image-to-video motion pilot using
   the approved image, e.g. a bounded Wan2.2 TI2V 5B ComfyUI job at
   realistically supported T4 resolution/VRAM. Test real character movement,
   head, posture and hands; not merely camera zoom. Try minimal quotas only.
   LTX-2.5 is NOT a safe default on Kaggle T4 16GB; official guidance is 32GB+.
6. Independently approve the pilot before extending it. Stable face/bones,
   costume, jaw, hands, physical coherence, natural expressions, no stutter.
   Speech lip-sync and jaw tracking are evaluated separately.
7. AFTER both approvals, animate the remaining scenes and reuse the existing
   genuine six Russian Qwen3 WAVs if their delivery is acceptable. Assemble
   the approved video clips with studio.hybrid_master.produce, subtitles,
   balanced soundtrack, final technical and creative QA. No InVideo.
8. The user explicitly approves any final before social posting. No automatic
   posting, no automatic full GPU rerender, no replacing this bad review by
   renaming the same seven images.

The old full GPU workflow is quarantined: automatic push trigger removed and
job disabled. It must not be re-enabled until a NEW identity-conditioned I2V
worker with approved source assets and verified motion is connected. The
quality gate checks hashes and file structure; human review supplies semantic
judgement. A boolean automatically set by the producer is not human approval.
