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
    "RAW ANALOG FILM PHOTOGRAPH, 35mm live-action movie still, real human actress, "
    "naturally elegant 31-year-old adult brunette funeral professional, "
    "LONG flowing dark chestnut WAVY HAIR, oval natural human face, "
    "subtle realistic skin texture with a tiny beauty mark below her left eye, "
    "dark-brown expressive almond-shaped eyes, finely defined eyebrows, "
    "fitted MATTE BLACK long-sleeve HIGH-NECK LACE blouse, "
    "tiny antique silver cameo necklace, holding a black pen at a walnut desk, "
    "subtle sarcastic expression, warm brass desk lamp, "
    "deep burgundy velvet curtains, atmospheric tasteful gothic office, "
    "cinematic natural proportions, authentic tungsten practical lighting, "
    "rich understated film color grading, photoreal detailed skin, "
    "subtle depth of field, shot on Kodak Vision3 500T, exceptional film photography."
)
NEGATIVE = (
    "illustration, cartoon, animation, comic, anime, painting, drawing, "
    "black ink outlines, plastic CGI, 3D render, stylized doll, airbrushed skin, "
    "flat vector, synthetic-looking face, short bob hairstyle, blonde, "
    "red jacket, red blouse, red clothes, white dress, skeleton, "
    "oversaturated, bad anatomy, extra fingers, missing hands, watermark, text"
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
        num_inference_steps=34, guidance_scale=4.6,
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
