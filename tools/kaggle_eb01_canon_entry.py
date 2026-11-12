"""Private Kaggle / SDXL V6 canonical character candidate producer.

ONE bounded image generation per character; no unrelated independent scenes,
no video, no TTS, no publishing. Full PNGs remain only in PRIVATE Kaggle output.
A downsampled (non-source) review image may be downloaded as a GitHub artifact.
"""
from __future__ import annotations
import base64
import gc
import hashlib
import json
import os
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

CONFIG_B64 = "__EB01_CANON_CONFIG_B64__"
ROOT = Path("/kaggle/temp/eb01_v6_canon")
WORK = Path("/kaggle/working")
ARCHIVE = WORK / "EB01_CANON_SOURCE_PRIVATE.zip"
CONTACT = WORK / "EB01_CANON_REVIEW_CONTACT.jpg"
RECEIPT = WORK / "EB01_CANON_JOB.json"
MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
WIDTH, HEIGHT = 704, 1216
NEGATIVE = ("ordinary man, flesh-faced man, human male head, beard, human skin on skull, "
            "black-and-white illustration, line drawing, sketch, pencil art, cartoon, "
            "white suit, white jacket, bright suit, badly drawn anatomy, missing bones, "
            "extra arms, extra hands, deformed jaw, inconsistent proportions, dislocated neck, "
            "text, letters, captions, watermark, brand, poor quality, low-res, "
            "oversaturated, plastic CGI, theatrical mask, halloween rubber face")
PROMPTS = {
    "KOSTYA": (
        "EXPOSED IVORY HUMAN SKELETON, fully visible skull made of real bone, "
        "open empty dark orbital eye sockets, complete visible teeth and movable lower bony jaw, "
        "fleshless bony hands articulated at the wrists and finger joints. "
        "The SKELETON is a theatrical gothic-comedy gentleman seated in an interview chair, "
        "wearing a perfectly fitted MATTE BLACK wool three-piece suit with a black vest, "
        "crisp WHITE shirt, narrow dark BURGUNDY tie, BLACK felt fedora. "
        "Bone skull prominent in the top half, both bony hands visible over a polished WALNUT DESK. "
        "Do not turn this skeleton into a man. Elegant burgundy velvet curtains behind him, "
        "an amber brass desk lamp illuminating anatomical ivory bone from the left. "
        "Beautiful premium 35mm cinematic photograph, film still by an acclaimed art director, "
        "expensive warm chiaroscuro, photorealistic natural depth, intricately detailed bone anatomy, "
        "full-color rich cinematic color grading, 85mm portrait lens, sharp detailed eyesockets, "
        "shallow depth of field, meticulously tailored BLACK fabric, believable physical materials."
    ),
    "ELIZABETH": (
        "ELIZABETH BONES, stylish 31-year-old adult funeral professional, elegant European brunette, "
        "recognizable natural oval face with small beauty mark just under her LEFT eye, "
        "almond-shaped dark-brown eyes, sharply defined expressive eyebrows and subtle smile, "
        "LONG DARK CHESTNUT WAVY HAIR falling to her chest, naturally textured fair skin, "
        "wearing a fitted MATTE BLACK long-sleeved high-neck lace blouse and a tiny silver cameo "
        "necklace. Sitting professionally behind a polished walnut interview desk, black pen in "
        "her right hand, cream resume on the desk, brass desk lamp at her left and "
        "burgundy velvet curtains in background. Not a different actress. "
        "Natural genuine human appearance, believable realistic posture and fabric, "
        "highly polished full-color 35mm photographic film still, superb skin texture, "
        "cinematic warm amber and wine-red lighting, 85mm cinematic portrait lens, "
        "restrained professional gothic elegance, beautiful photography, "
        "soft depth of field, realistic facial detail, sharp expressive eyes."
    ),
}
NEGATIVE_ELIZABETH = (
    "different person, skeleton, skull face, white dress, white blouse, pink shirt, "
    "short hair, bob hairstyle, blonde, redhead, fantasy costume, old woman, doll, "
    "plastic skin, cartoon, painting, drawing, unrealistic anatomy, extra hands, "
    "staring blank face, low resolution, blurry, watermark, letters, text, caption"
)

def cmd(*args, timeout=1500):
    subprocess.run(args, check=True, timeout=timeout)

def create_art(cfg):
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    from PIL import Image, ImageDraw, ImageStat
    if not torch.cuda.is_available():
        raise RuntimeError("Private 16GB NVIDIA GPU not available")
    device = torch.cuda.get_device_properties(0)
    if device.total_memory < 14 * (2**30):
        raise RuntimeError("GPU must have ~16 GB memory")
    ROOT.mkdir(parents=True, exist_ok=False)
    print("TW_EB01_CANON_ONE_PASS_GPU",torch.cuda.get_device_name(0),flush=True)
    cmd(sys.executable,"-m","pip","install","-q","diffusers>=0.35,<0.38",
        "transformers>=4.55,<5","accelerate>=1,<2","Pillow>=10,<12",
        "safetensors>=0.4",timeout=1200)
    pipe=StableDiffusionXLPipeline.from_pretrained(
        MODEL,torch_dtype=torch.float16,variant="fp16",use_safetensors=True)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,algorithm_type="dpmsolver++",solver_order=2,use_karras_sigmas=True)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    records={}
    for name, seed in (("KOSTYA",cfg["seeds"]["KOSTYA"]),
                       ("ELIZABETH",cfg["seeds"]["ELIZABETH"])):
        image=pipe(
            prompt=PROMPTS[name],
            negative_prompt=NEGATIVE if name=="KOSTYA" else NEGATIVE_ELIZABETH,
            height=HEIGHT,width=WIDTH,num_inference_steps=36,
            guidance_scale=6.6,
            generator=torch.Generator(device="cpu").manual_seed(seed)
        ).images[0].convert("RGB")
        if image.size!=(WIDTH,HEIGHT) or max(ImageStat.Stat(image).stddev)<12:
            raise RuntimeError("Invalid/blank canonical candidate for "+name)
        path=ROOT/(name+".png")
        image.save(path,optimize=True)
        if path.stat().st_size<100000:
            raise RuntimeError("Suspiciously small candidate "+name)
        records[name]={
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "size":path.stat().st_size,"width":WIDTH,"height":HEIGHT,
            "model":MODEL,"seed":seed,"art_reviewed":False
        }
        print("EB01_CANON_GENERATED_PENDING_REVIEW",name,path.stat().st_size,flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # NON-SOURCE downsampled reviewer proof. Original PNGs stay in private Kaggle.
    thumb_w,thumb_h=242,418
    contact=Image.new("RGB",(thumb_w*2,thumb_h+65),(29,22,31))
    draw=ImageDraw.Draw(contact)
    for i,name in enumerate(("KOSTYA","ELIZABETH")):
        with Image.open(ROOT/(name+".png")) as p:
            p.thumbnail((thumb_w,thumb_h),Image.Resampling.LANCZOS)
            contact.paste(p,((thumb_w-p.width)//2+thumb_w*i,0))
        draw.text((thumb_w*i+12,thumb_h+7),name,fill="white")
        draw.text((thumb_w*i+12,thumb_h+27),"UNAPPROVED ART REVIEW",fill="#ffaaaa")
    contact.save(CONTACT,quality=81,optimize=True)
    with zipfile.ZipFile(ARCHIVE,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("KOSTYA","ELIZABETH"):
            z.write(ROOT/(name+".png"),arcname="canon/"+name+".png")
    receipt={
        "schema":1,"nonce":cfg["nonce"],"success":True,"model":MODEL,
        "source_revision":cfg["source_revision"],"private":True,
        "generated_images":records,
        "archive_sha256":hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "contact_sha256":hashlib.sha256(CONTACT.read_bytes()).hexdigest(),
        "model_downloaded_for_run":True,"gpu_inference_performed":True,
        "video_generated":False,"image_identity_verified":False,
        "human_visual_review_required":True,"human_visual_review":False,
        "full_video_allowed":False,"published":False}
    RECEIPT.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print("EB01_CANON_TWO_REAL_IMAGES_PRIVATE_PENDING_REVIEW",flush=True)

def main():
    cfg=json.loads(base64.b64decode(CONFIG_B64,validate=True))
    try:
        assert cfg["schema"]==1 and cfg["model"]==MODEL
        assert cfg["max_images"]==2 and cfg["allow_video"] is False
        assert len(cfg["seeds"])==2
        create_art(cfg)
    except BaseException as exc:
        RECEIPT.write_text(json.dumps({
            "schema":1,"nonce":cfg.get("nonce"),"success":False,
            "error_type":type(exc).__name__,"error_summary":str(exc)[:280],
            "video_generated":False,"published":False
        },indent=2)+"\n")
        print("EB01_CANON_ERROR",type(exc).__name__,str(exc)[:280],flush=True)
        traceback.print_exc()
        raise

if __name__=="__main__":
    main()
