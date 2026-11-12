"""ONE native Wan2.2 motion preview from PRIVATE Kaggle canon, no public art.

This packager runs only after canonical stills exist in the user's private
Kaggle kernel. It does NOT generate seven shots, mark artwork approved,
overwrite private sources, or use an external editor.

Source worker uses the official ComfyUI Wan2.2 5B API integration that was
independently proven by the user's GPU run 42980917354 (17 real frames).
This pilot takes 49 native frames (2.04 seconds at 24 fps).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

from studio.eb01_canon_kaggle import MIXED_MODEL
from studio.eb01_canon_kaggle import ARCHIVE as CANON_ARCHIVE
from studio.eb01_canon_kaggle import RECEIPT as CANON_RECEIPT
from studio.kaggle_neural_video import _bundle
from studio.kaggle_api import ROOT
from studio.kaggle_identity import HANDLE
from studio.release_gate import sha256

WORKER="tools/kaggle_wan22_video_entry.py"
FRAMES=49
STEPS=20
WIDTH,HEIGHT=448,800
RUNNER_OUTPUT="EB01_KOSTYA_REAL_WAN_I2V_PREVIEW.mp4"
PROMPT=(
    "Cinematic moving film close-up, locked camera. The SAME KOSTYA from the first "
    "reference image is a fully fleshless adult human SKELETON: ivory bone skull, "
    "empty dark eye sockets, visible individual teeth and articulated bony fingers. "
    "He wears exactly the same MATTE BLACK tailored wool suit and small black bow tie "
    "as pictured, with NO hat, NO necktie and NO new accessories. "
    "He sits on the same dark red upholstered antique chair in the same richly "
    "lit room, the exact same skeletal hands remain visible. "
    "The anatomical skeleton slightly turns his skull to his left in a natural "
    "physical rotation, lower jaw opens and closes naturally as if saying a short "
    "line, the neck and shoulders shift with natural body movement and a bony "
    "hand makes a small restrained gesture by the chair. Beautiful rich amber and "
    "burgundy cinematic lighting, expensive photorealism, coherent black clothing, "
    "physically correct skull and jaw throughout; ABSOLUTELY no human flesh "
    "on skull, no living man, no changing costume, no morphing or trembling."
)

def parent_reference(folder: Path, who: str = "KOSTYA") -> tuple[bytes,str]:
    root=Path(folder)
    receipts=list(root.rglob(CANON_RECEIPT))
    archives=list(root.rglob(CANON_ARCHIVE))
    if len(receipts)!=1 or len(archives)!=1:
        raise FileNotFoundError("Exact private canon source archive and receipt required")
    data=json.loads(receipts[0].read_text(encoding="utf-8"))
    if not (data.get("success") is True
            and data.get("private") is True
            and data.get("gpu_inference_performed") is True
            and data.get("published") is False
            and data.get("model") in {"stabilityai/stable-diffusion-xl-base-1.0", MIXED_MODEL}):
        raise ValueError("Only real PRIVATE original model artwork can seed the motion proof")
    if sha256(archives[0])!=data.get("archive_sha256"):
        raise ValueError("Private source archive corrupt")
    with zipfile.ZipFile(archives[0]) as z:
        art=z.read("canon/"+who+".png")
    expected=data["generated_images"][who]["sha256"]
    if hashlib.sha256(art).hexdigest()!=expected or len(art)<100000:
        raise ValueError("Kostya original private source not pinned")
    with Image.open(io.BytesIO(art)) as original:
        if original.size != (704,1216):
            raise ValueError("Kostya original dimensions invalid")
        shot=ImageOps.fit(original.convert("RGB"),(WIDTH,HEIGHT),
                          method=Image.Resampling.LANCZOS,centering=(.5,.36))
        raw=io.BytesIO()
        shot.save(raw,format="PNG")
    return raw.getvalue(),expected

def prepare(username: str, nonce: str, source_output: Path,
            destination: Path, *,root: Path=ROOT) -> dict:
    if not HANDLE.fullmatch(username) or not re.fullmatch(r"[A-Za-z0-9_-]{6,90}",nonce):
        raise ValueError("Invalid private account/job nonce")
    raw,canon_sha=parent_reference(source_output)
    reference_sha=hashlib.sha256(raw).hexdigest()
    fullroot=Path(root).resolve()
    source=(fullroot/WORKER).read_text(encoding="utf-8")
    if (source.count("__TW_VIDEO_CONFIG__")!=1 or
            source.count("__TW_VIDEO_BUNDLE__")!=1 or
            source.count("draw_fixture(reference)")!=1 or
            source.count('data.get("frames") == 17')!=1):
        raise ValueError("Pinned Wan2.2 private worker source drifted; stop rather than guessing")
    payload={
        "schema":1,"nonce":nonce,"source_revision":os.environ.get("GITHUB_SHA","local-test-revision"),
        "model":"wan22-5b","width":WIDTH,"height":HEIGHT,
        "frames":FRAMES,"steps":STEPS,"seed":20261112,
        "prompt":PROMPT,"reference_png_b64":base64.b64encode(raw).decode("ascii"),
        "reference_sha256":reference_sha,
        "original_canon_sha256":canon_sha,
        "source_art_private":True,
        "artistic_review_required":True,"allow_full_episode":False}
    injection=(
        "reference.write_bytes(base64.b64decode(data['reference_png_b64'],validate=True))\n"
        "    if hashlib.sha256(reference.read_bytes()).hexdigest()!=data['reference_sha256']:\n"
        "        raise RuntimeError('Private canonical reference was modified in worker')\n"
    )
    rendered=source.replace("draw_fixture(reference)",injection)
    rendered=rendered.replace('data.get("frames") == 17','data.get("frames") == 49')
    rendered=rendered.replace(
        "__TW_VIDEO_CONFIG__",
        base64.b64encode(json.dumps(payload,sort_keys=True).encode()).decode())
    rendered=rendered.replace("__TW_VIDEO_BUNDLE__",base64.b64encode(_bundle(fullroot)).decode())
    assert all(s not in rendered for s in ("__TW_VIDEO_CONFIG__","__TW_VIDEO_BUNDLE__"))
    slug="eb01-v6-real-kostya-"+hashlib.sha256(nonce.encode()).hexdigest()[:12]
    metadata={
        "id":username+"/"+slug, "title":slug,
        "code_file":"kernel_main.py", "language":"python", "kernel_type":"script",
        "is_private":"true","enable_gpu":"true","enable_internet":"true",
        "machine_shape":"NvidiaTeslaT4","dataset_sources":[],
        "competition_sources":[],"kernel_sources":[],"model_sources":[]}
    dest=Path(destination)
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"kernel_main.py").write_text(rendered,encoding="utf-8")
    (dest/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return {
        "status":"ONE_PRIVATE_WAN_MOTION_PROOF_READY",
        "kernel":metadata["id"],"reference_sha256":reference_sha,
        "original_canon_sha256":canon_sha,"frame_count":FRAMES,
        "gpu_executed":False,"review_required":True,"published":False}

def verify(folder: Path, nonce: str, parent_original_sha: str) -> dict:
    from studio.kaggle_neural_video import VIDEO_ARCHIVE, RECEIPT
    root=Path(folder)
    receipts=list(root.rglob(RECEIPT)); archives=list(root.rglob(VIDEO_ARCHIVE))
    if len(receipts)!=1 or len(archives)!=1:
        raise FileNotFoundError("Missing private Wan actual MP4 archive + receipt")
    data=json.loads(receipts[0].read_text(encoding="utf-8"))
    if not (data.get("success") is True and data.get("nonce")==nonce
            and data.get("model")=="wan22-5b"
            and data.get("inference_performed") is True
            and data.get("video",{}).get("frames")==FRAMES
            and data.get("video",{}).get("width")==WIDTH
            and data.get("video",{}).get("height")==HEIGHT
            and data.get("published") is False):
        raise ValueError("Not a real 49-frame Wan I2V artifact")
    if sha256(archives[0])!=data.get("archive_sha256"):
        raise ValueError("Private Wan archive SHA-256 mismatch")
    with zipfile.ZipFile(archives[0]) as z:
        if set(z.namelist())!={"neural_proof.mp4","neural_proof.receipt.json","technical_qc.json"}:
            raise ValueError("Unexpected Wan private GPU result files")
        clip=z.read("neural_proof.mp4")
        proof=json.loads(z.read("neural_proof.receipt.json"))
        qc=json.loads(z.read("technical_qc.json"))
    if (not clip[:64].count(b"ftyp")
            or hashlib.sha256(clip).hexdigest()!=data.get("clip_sha256")
            or proof.get("model")!="wan22-5b"
            or proof.get("inference_performed") is not True
            or qc.get("all_frames_decoded") is not True
            or qc.get("real_motion_detected") is not True
            or float(qc.get("first_last_mean_difference",0))<.75
            or not re.fullmatch("[a-f0-9]{64}",str(proof.get("reference_sha256")))):
        raise ValueError("No validated real moving skeleton reference clip")
    if not re.fullmatch("[a-f0-9]{64}",parent_original_sha):
        raise ValueError("Missing private parent canon hash")
    return {
        "status":"GENUINE_KOSTYA_WAN_I2V_MOTION_PROOF_REVIEW_REQUIRED",
        "nonce":nonce,"frames":FRAMES,"seconds":data["video"].get("seconds"),
        "model":"wan22-5b","source_parent_sha256":parent_original_sha,
        "reference_sha256":proof["reference_sha256"],
        "video_sha256":data["clip_sha256"],
        "first_last_mean_difference":qc["first_last_mean_difference"],
        "human_approved":False,"published":False}

def main(argv=None):
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("prepare")
    a.add_argument("--username",required=True)
    a.add_argument("--nonce",required=True)
    a.add_argument("--source-output",type=Path,required=True)
    a.add_argument("--destination",type=Path,required=True)
    b=sub.add_parser("verify")
    b.add_argument("--path",type=Path,required=True)
    b.add_argument("--nonce",required=True)
    b.add_argument("--parent-original-sha",required=True)
    x=p.parse_args(argv)
    result=prepare(x.username,x.nonce,x.source_output,x.destination) if x.action=="prepare" else verify(x.path,x.nonce,x.parent_original_sha)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
