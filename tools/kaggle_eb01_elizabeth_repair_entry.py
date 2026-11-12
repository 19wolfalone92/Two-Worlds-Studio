"""Private ONE-image photoreal Elizabeth correction. Reuses Kostya byte-identically.

Original KOSTYA source is embedded only in the dynamically generated PRIVATE
Kaggle notebook by the GitHub runner, not in this public file. No video/TTS.
"""
from __future__ import annotations
import base64
import gc
import hashlib
import io
import json
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

CONFIG_B64="__EB01_ELIZABETH_REPAIR_CONFIG_B64__"
MODEL="RunDiffusion/Juggernaut-XL-v9"
RECEIPT_MODEL="eb01-v6-mixed-sdxl-kostya-juggernaut-v9-elizabeth"
PARENT_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
WORK=Path("/kaggle/working")
PRIVATE=Path("/kaggle/temp/eb01_v6_repaired")
ARCHIVE=WORK/"EB01_CANON_SOURCE_PRIVATE.zip"
RECEIPT=WORK/"EB01_CANON_JOB.json"
CONTACT=WORK/"EB01_CANON_REVIEW_CONTACT.jpg"
WIDTH,HEIGHT=704,1216
PROMPT=(
    "RAW ANALOG FILM PHOTOGRAPH, photorealistic high-end 35mm film portrait, "
    "one naturally beautiful 31-year-old brunette funeral interview officer, "
    "LONG flowing DARK CHESTNUT WAVY HAIR, honest natural human face with "
    "subtle freckles and a small beauty mark below her left eye, "
    "soft oval face, warm dark-brown almond eyes, dark eyebrows, pale textured skin, "
    "wearing an elegant FITTED BLACK LONG-SLEEVE HIGH-NECK LACE BLOUSE and "
    "a small silver cameo necklace. Seated at a walnut desk, holding a black pen, "
    "warm brass reading lamp, deep burgundy curtains, tasteful Gothic office, "
    "live-action cinema screenshot, exquisite real-world details, realistic lens blur, "
    "professional photograph shot on Kodak Vision3 500T, natural rich shadows."
)
NEGATIVE=(
    "illustration, cartoon, animation, cel shading, anime, graphic novel, vector, "
    "drawing, painting, digital painting, black ink outline, airbrushed face, "
    "CGI, 3D render, doll, plastic wax skin, flat gray background, red blouse, "
    "red jacket, red clothes, white dress, bob hairstyle, short hair, blonde, "
    "heavy theatrical makeup, distorted fingers, extra hands, badly drawn eyes, "
    "letters, captions, watermark, text"
)

def generate(data):
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<14*2**30:
        raise RuntimeError("Expected 16GB-class private NVIDIA T4 for photoreal image repair")
    PRIVATE.mkdir(parents=True,exist_ok=False)
    # Validate original KOSTYA before ANY model download/GPU generation.
    from PIL import Image,ImageStat,ImageDraw
    kostya_bytes=base64.b64decode(data["kostya_png_b64"],validate=True)
    if hashlib.sha256(kostya_bytes).hexdigest()!=data["kostya_sha256"]:
        raise RuntimeError("Canonical skeleton SHA changed during transfer")
    with Image.open(io.BytesIO(kostya_bytes)) as img:
        if img.format!="PNG" or img.size!=(WIDTH,HEIGHT):
            raise RuntimeError("Canonical skeleton PNG geometry or encoding changed")
    (PRIVATE/"KOSTYA.png").write_bytes(kostya_bytes)
    print("PRESERVED_KOSTYA_CANON_WITHOUT_REGEN",len(kostya_bytes),flush=True)

    subprocess.run([
        sys.executable,"-m","pip","install","-q",
        "diffusers>=0.35,<0.38","transformers>=4.55,<5",
        "accelerate>=1,<2","Pillow>=10,<12","safetensors>=0.4"
    ],check=True,timeout=1500)

    from diffusers import StableDiffusionXLPipeline,DPMSolverMultistepScheduler
    pipe=StableDiffusionXLPipeline.from_pretrained(
        MODEL,torch_dtype=torch.float16,variant="fp16",use_safetensors=True)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,algorithm_type="dpmsolver++",solver_order=2,
        use_karras_sigmas=True)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling();pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    image=pipe(
        prompt=PROMPT,negative_prompt=NEGATIVE,
        width=WIDTH,height=HEIGHT,num_inference_steps=35,
        guidance_scale=4.6,
        generator=torch.Generator(device="cpu").manual_seed(data["elizabeth_seed"]),
    ).images[0].convert("RGB")
    dest=PRIVATE/"ELIZABETH.png"
    image.save(dest,optimize=True)
    if dest.stat().st_size<120000 or max(ImageStat.Stat(image).stddev)<11:
        raise RuntimeError("Photo repair missing or blank")
    del pipe
    gc.collect();torch.cuda.empty_cache()
    if dest.read_bytes()==kostya_bytes:
        raise RuntimeError("Elizabeth repair returned original skeleton, not a woman")

    # small non-source review image; original remains PRIVATE only
    cols,rows=242,418
    sheet=Image.new("RGB",(cols*2,rows+65),(25,22,31))
    draw=ImageDraw.Draw(sheet)
    for i,who in enumerate(("KOSTYA","ELIZABETH")):
        with Image.open(PRIVATE/(who+".png")) as im:
            im=im.convert("RGB")
            im.thumbnail((cols,rows),Image.Resampling.LANCZOS)
            sheet.paste(im,(cols*i+(cols-im.width)//2,0))
        draw.text((cols*i+10,rows+7),who,fill="white")
        draw.text((cols*i+10,rows+28),"UNAPPROVED ART REVIEW",fill="#ffaaaa")
    sheet.save(CONTACT,format="JPEG",quality=82,optimize=True)
    with zipfile.ZipFile(ARCHIVE,"w",zipfile.ZIP_DEFLATED) as zipped:
        for who in ("KOSTYA","ELIZABETH"):
            zipped.write(PRIVATE/(who+".png"),arcname="canon/"+who+".png")
    elizabeth_bytes=dest.read_bytes()
    models={
        "KOSTYA":{"sha256":data["kostya_sha256"],"size":len(kostya_bytes),
                  "width":WIDTH,"height":HEIGHT,
                  "model":PARENT_MODEL,"reused_unchanged":True,
                  "art_reviewed":False},
        "ELIZABETH":{"sha256":hashlib.sha256(elizabeth_bytes).hexdigest(),
                     "size":len(elizabeth_bytes),"width":WIDTH,"height":HEIGHT,
                     "model":MODEL,"seed":data["elizabeth_seed"],"reused_unchanged":False,
                     "art_reviewed":False}
    }
    report={
        "schema":1,"nonce":data["nonce"],"success":True,"private":True,
        "source_revision":data["source_revision"],"model":RECEIPT_MODEL,
        "parent_KOSTYA_sha256":data["kostya_sha256"],
        "generated_images":models,
        "archive_sha256":hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "contact_sha256":hashlib.sha256(CONTACT.read_bytes()).hexdigest(),
        "gpu_inference_performed":True,"image_identity_verified":False,
        "human_visual_review":False,"human_visual_review_required":True,
        "full_video_allowed":False,"video_generated":False,
        "published":False}
    RECEIPT.write_text(json.dumps(report,indent=2)+"\n")
    print("EB01_JUGGERNAUT_ONE_PHOTOREAL_REPAIR_GENERATED_PRIVATE",dest.stat().st_size,flush=True)

def main():
    data=json.loads(base64.b64decode(CONFIG_B64,validate=True))
    try:
        if not(data.get("schema")==1 and data.get("model")==MODEL
               and data.get("allow_video") is False
               and data.get("max_new_images")==1 and data.get("reused_kostya") is True):
            raise ValueError("Unreviewed or quota-unsafe photoreal repair job")
        generate(data)
    except BaseException as exc:
        RECEIPT.write_text(json.dumps({
            "schema":1,"nonce":data.get("nonce"),"success":False,
            "error_type":type(exc).__name__,"error_summary":str(exc)[:300],
            "video_generated":False,"published":False},indent=2)+"\n")
        print("EB01_ELIZABETH_PHOTOREAL_REPAIR_FAILED",type(exc).__name__,str(exc)[:200],flush=True)
        traceback.print_exc()
        raise

if __name__=="__main__":
    main()
