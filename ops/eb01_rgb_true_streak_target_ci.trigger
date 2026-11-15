2026-11-15 target real FFmpeg H264 blue palette versus 2px stripe; review statuses remain fail-closed, Kaggle GPU inference successful but color review-blocked not false FP16 crash, FP32 video VAE, public GitHub console must not leak generated prompt filenames. No GPU.

2026-11-15 observed EXACT H264 yuv420 2 original blue streak rows become 4 adjacent decoded chroma rows. New code groups contiguous long-chroma rows and accepts only thin <=6px groups as line candidate; larger static blue clothes wall remain REVIEW not clean. Run actual FFmpeg tests 29+, do not start GPU.
