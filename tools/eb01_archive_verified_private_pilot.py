"""Archive ONE real SHA-locked neural video to OWNER-PRIVATE Kaggle dataset.

Only owner KAGGLE_API_TOKEN. Never to PUBLIC GitHub artifacts. Kaggle CLI
stdout/stderr suppressed because they may include private media filenames.
Kaggle datasets create is private by default; NEVER use --public.
Independent metadata check required before claiming remote ACL verification.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re,shutil,subprocess,tempfile
from pathlib import Path
from tools.eb01_gpu_watchdog import validate_actual_pilot

KERNEL_RE=re.compile(r"^wolf19loner92/eb01-ltx2b-pilot33-[0-9a-f]{12}$")
HEX64=re.compile(r"^[0-9a-f]{64}$")

class PrivateArchiveBlocked(ValueError): pass

def safe_cmd(argv,timeout=180):
    if "--public" in argv:
        raise PrivateArchiveBlocked("Never publish original actor video")
    try:
        p=subprocess.run(argv,stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         timeout=timeout,check=False)
        return p.returncode==0
    except (OSError,subprocess.TimeoutExpired):
        return False

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def private_dataset_meta(kernel, expected):
    if not KERNEL_RE.fullmatch(kernel) or not HEX64.fullmatch(expected):
        raise PrivateArchiveBlocked("Unpinned source model or original hash")
    suffix=kernel.rsplit("-",1)[-1]
    return {"title":"EB01 Owner Private Native Video "+suffix,
        "id":"wolf19loner92/eb01-owner-private-motion-"+suffix,
        "subtitle":"Private original actor review movie and proof only",
        "description":"Original owner-private 33-frame AI motion preview; not approved, "
           "not for public redistribution, training, or media publication. "
           "Original characters and creative rights reserved.",
        "licenses":[{"name":"copyright-authors"}],"isPrivate":True}

def archive(kernel,expected,temp_parent):
    if not KERNEL_RE.fullmatch(kernel) or not HEX64.fullmatch(expected):
        raise PrivateArchiveBlocked("Kaggle source and SHA not pinned")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise PrivateArchiveBlocked("Owner credential unavailable")
    report={"schema":1,"gpu_restarted":False,"published":False,
        "full_movie_ready":False,"public_github_media_upload":False,
        "owner_original_mp4_sha256":expected,"independent_remote_acl":False}
    with tempfile.TemporaryDirectory(prefix="eb01-private-",dir=str(temp_parent)) as d:
        root=Path(d)
        output=root/"model_private_output"
        output.mkdir(mode=0o700)
        if not safe_cmd(["kaggle","kernels","output",kernel,
                         "-p",str(output),"--force"],timeout=240):
            report["status"]="KAGGLE_PRIVATE_ORIGINAL_UNAVAILABLE"
            return report
        video=output/"S02_PILOT33.mp4"
        receipt=output/"EB01_S02_LTX_PILOT33_JOB.json"
        if not video.is_file() or not receipt.is_file() or sha(video)!=expected:
            report["status"]="REAL_ORIGINAL_VIDEO_OR_SHA_UNVERIFIED"
            return report
        source=validate_actual_pilot(output)
        if (source.get("original_pilot_sha256")!=expected or
            source.get("decoded_true_neural_frames")!=33 or
            source.get("artifact_scan_status")!="NO_NEON_ARTIFACT_FOUND" or
            source.get("full_human_artistic_approval") is not False):
            report["status"]="ORIGINAL_COLOR_OR_NATIVE_PROOF_MISSING"
            return report
        original=json.loads(receipt.read_text(encoding="utf-8"))
        if original.get("success") is not True or original.get("clip_sha256")!=expected:
            report["status"]="ORIGINAL_RECEIPT_INVALID"
            return report
        safe={k:original.get(k) for k in (
            "schema","shot","success","status","model","mode","frame_count",
            "fps","actual_gpu_seconds","width","height","clip_sha256",
            "true_gpu_neural_inference","human_artistic_approved",
            "whole_movie_ready","published",
            "audio_conditioned_lipsync_verified")}
        if safe.get("whole_movie_ready") is not False:
            report["status"]="MUST_NOT_ARCHIVE_MISLABELED_MOVIE"
            return report
        stage=root/"owner_private_dataset"
        stage.mkdir(mode=0o700)
        meta=private_dataset_meta(kernel,expected)
        (stage/"dataset-metadata.json").write_text(
            json.dumps(meta,indent=2)+"\n",encoding="utf-8")
        shutil.copy2(video,stage/"S02_ORIGINAL_33F_OWNER_REVIEW.mp4")
        (stage/"ORIGINAL_RECEIPT_SAFE.json").write_text(
            json.dumps(safe,indent=2)+"\n",encoding="utf-8")
        if not safe_cmd(["kaggle","datasets","create","-p",str(stage),
                         "--quiet","--dir-mode","skip"],timeout=300):
            report["status"]="PRIVATE_KAGGLE_DATASET_CREATE_FAILED"
            return report
        report["created_private_default"]=True
        meta_dir=root/"owner_metadata"
        meta_dir.mkdir(mode=0o700)
        if safe_cmd(["kaggle","datasets","metadata",meta["id"],
                     "-p",str(meta_dir)],timeout=100):
            remote_file=meta_dir/"dataset-metadata.json"
            if remote_file.is_file():
                try:remote=json.loads(remote_file.read_text())
                except (ValueError,UnicodeError):remote={}
                if remote.get("isPrivate") is True:
                    report["independent_remote_acl"]=True
                elif remote.get("isPrivate") is False:
                    # On detected privacy conflict, attempt immediate revocation.
                    report["delete_attempted"]=safe_cmd(
                        ["kaggle","datasets","delete",meta["id"],"--yes"],
                        timeout=100)
                    report["status"]="REMOTE_DATASET_PRIVACY_CONFLICT"
                    return report
        report["status"]=("OWNER_PRIVATE_DATASET_CREATED_ACL_VERIFIED"
            if report["independent_remote_acl"] else
            "OWNER_PRIVATE_DATASET_CREATED_ACL_NOT_INDEPENDENTLY_VERIFIED")
        report["owner_private_kaggle_dataset"]=meta["id"]
        report["original_video_byte_count"]=video.stat().st_size
        return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--kernel",required=True)
    p.add_argument("--sha256",required=True)
    args=p.parse_args()
    report=archive(args.kernel,args.sha256,Path(os.environ["RUNNER_TEMP"]))
    out=Path(os.environ["RUNNER_TEMP"])/"EB01_PRIVATE_ARCHIVE_STATUS_ONLY.json"
    out.write_text(json.dumps(report,indent=2)+"\n")
    print("EB01_OWNER_PRIVATE_VIDEO_ARCHIVE",json.dumps(report,sort_keys=True))
    return 0 if report["status"].startswith("OWNER_PRIVATE_DATASET_CREATED") else 2

if __name__=="__main__":
    raise SystemExit(main())
