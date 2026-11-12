"""Generate ONE private photo-real Elizabeth candidate, without embedded PNGs.

Kaggle SaveKernel returned HTTP 400 for the previous submission that nested
the already-generated skeleton PNG in two layers of base64 source code.
Kostya is retained in an earlier private Kaggle output, referenced ONLY by
SHA256; he is NOT uploaded or generated again. The repaired Elizabeth stays
in a distinct PRIVATE output. A later read-only collector joins provenance.
"""
from __future__ import annotations

import base64
import gc
import hashlib
import io
import json
import re
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

CONFIG_B64 = "__EB01_ELIZABETH_REPAIR_CONFIG_B64__"
MODEL = "RunDiffusion/Juggernaut-XL-v9"
SOURCE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
WORK = Path("/kaggle/working")
PRIVATE = Path("/kaggle/temp/eb01_v6_elizabeth_photo")
ARCHIVE = WORK / "EB01_ELIZABETH_PRIVATE_ONLY.zip"
RECEIPT = WORK / "EB01_ELIZABETH_REPAIR_JOB.json"
PREVIEW = WORK / "EB01_ELIZABETH_REVIEW_LOW_RES.jpg"
WIDTH, HEIGHT = 704, 1216
PROMPT = (
    "CINEMATIC MEDIUM-WIDE FILM STILL, NOT A CLOSE UP, camera sees entire UPPER "
    "BODY from WAIST to head AND DESK in the bottom third. A real 31-year-old "
    "brunette actress ELIZABETH BONES sits behind a polished WALNUT INTERVIEW DESK "
    "in an atmospheric Victorian gothic office. She wears an unmistakably "
    "JET BLACK FITTED HIGH-NECK LONG-SLEEVE LACE BLOUSE, the black long "
    "sleeves and BLACK LACE torso are clearly visible and occupy the central "
    "and lower part of the image. No white fabric, no exposed shoulders. "
    "LONG WAVY DARK CHESTNUT HAIR falling to her chest, naturally attractive "
    "oval face, expressive BROWN eyes, tiny left-eye beauty mark, authentic "
    "real skin texture, subtle skeptical interviewing expression. Her right "
    "hand holds a dark pen against a cream resume on the walnut desk. A warm "
    "BRASS DESK LAMP, BURGUNDY VELVET DRAPES and dark walnut wood-panelled "
    "walls are distinctly visible behind her. CINEMA PRODUCTION DESIGN, "
    "beautiful moody warm amber practical lighting, dark rich professional "
    "color grade, Kodak Vision3 tungsten 35mm FILM, natural skin "
    "with no glamour retouching, expensive photoreal live action, "
    "50mm cinema lens, art-directed composition and real physical materials."
)
NEGATIVE = (
    "tight headshot, selfie, cropped shoulders, portrait with no torso, "
    "modern bright white bedroom, white wall, grey wall, white top, "
    "WHITE BLOUSE, beige top, white sleeveless dress, bare shoulders, "
    "light clothing, bright pale outfit, red blouse, red jacket, "
    "blonde, bob haircut, cartoon, comic book, ink lines, anime, painting, "
    "stylized 3D render, fake plastic doll, theatrical cosplay, "
    "muddy shapes, extra limbs, missing hands, watermark, letters, text"
)

def generate(data):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No private NVIDIA GPU available")
    if torch.cuda.get_device_properties(0).total_memory < 14 * (2**30):
        raise RuntimeError("Expected ~16 GB class GPU for SDXL candidate")
    PRIVATE.mkdir(parents=True, exist_ok=False)
    if not re.fullmatch("[a-f0-9]{64}", data["parent_KOSTYA_sha256"]):
        raise ValueError("Parent skeleton SHA must be pinned before generating Elizabeth")

    # Only required image generator dependencies are installed on the GPU host.
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "diffusers>=0.35,<0.38", "transformers>=4.55,<5",
        "accelerate>=1,<2", "Pillow>=10,<12", "safetensors>=0.4"
    ], check=True, timeout=1200)

    from PIL import Image, ImageDraw, ImageStat
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++",
        solver_order=2, use_karras_sigmas=True)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)

    image = pipe(
        prompt=PROMPT, negative_prompt=NEGATIVE, width=WIDTH, height=HEIGHT,
        num_inference_steps=38, guidance_scale=7.4,
        generator=torch.Generator(device="cpu").manual_seed(data["elizabeth_seed"]),
    ).images[0].convert("RGB")
    dest = PRIVATE / "ELIZABETH.png"
    image.save(dest, optimize=True)
    if (image.size != (WIDTH, HEIGHT) or dest.stat().st_size < 120000
            or max(ImageStat.Stat(image).stddev) < 11):
        raise RuntimeError("Image quality or PNG integrity is invalid")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # Only her ORIGINAL PNG is kept in this private Kaggle notebook output.
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(dest, "canon/ELIZABETH.png")

    # A small non-source thumbnail may be shared for human review.
    sheet = Image.new("RGB", (272, 520), (22, 17, 27))
    thumb = image.copy()
    thumb.thumbnail((272, 468), Image.Resampling.LANCZOS)
    sheet.paste(thumb, ((272-thumb.width)//2, 0))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 473), "ELIZABETH / PHOTO CANDIDATE", fill="#f6ede1")
    draw.text((12, 495), "UNAPPROVED - ORIGINAL PRIVATE", fill="#eaa9a9")
    sheet.save(PREVIEW, "JPEG", quality=83, optimize=True)

    photo = dest.read_bytes()
    report = {
        "schema": 2, "nonce": data["nonce"], "success": True,
        "private": True, "model": MODEL,
        "source_revision": data["source_revision"],
        "parent_source_kernel": data["parent_source_kernel"],
        "parent_KOSTYA_sha256": data["parent_KOSTYA_sha256"],
        "parent_KOSTYA_source_model": SOURCE_MODEL,
        "original_KOSTYA_untouched": True,
        "generated_images": {
            "ELIZABETH": {
                "sha256": hashlib.sha256(photo).hexdigest(),
                "size": len(photo), "width": WIDTH, "height": HEIGHT,
                "model": MODEL, "seed": data["elizabeth_seed"],
                "art_reviewed": False
            }
        },
        "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "preview_sha256": hashlib.sha256(PREVIEW.read_bytes()).hexdigest(),
        "gpu_inference_performed": True, "image_identity_verified": False,
        "human_visual_review": False, "full_video_allowed": False,
        "video_generated": False, "published": False
    }
    RECEIPT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("EB01_PRIVATE_ONE_PHOTOGRAPH_SUCCESS", len(photo),
          report["generated_images"]["ELIZABETH"]["sha256"], flush=True)

def main():
    data = json.loads(base64.b64decode(CONFIG_B64, validate=True))
    try:
        if not (data.get("schema") == 2 and data.get("model") == MODEL
                and data.get("max_new_images") == 1
                and data.get("reused_kostya") is True
                and data.get("allow_video") is False
                and "kostya_png_b64" not in data):
            raise ValueError("Malformed or oversized image-only production request")
        generate(data)
    except BaseException as exc:
        RECEIPT.write_text(json.dumps({
            "schema": 2, "nonce": data.get("nonce"), "success": False,
            "error_type": type(exc).__name__, "error_summary": str(exc)[:300],
            "video_generated": False, "published": False
        }, indent=2) + "\n")
        print("EB01_PHOTOREAL_FAILED", type(exc).__name__, str(exc)[:250], flush=True)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
