"""Archive ONE real SHA-locked neural video to OWNER-PRIVATE Kaggle dataset.

Only owner KAGGLE_API_TOKEN. Never to PUBLIC GitHub artifacts. Kaggle CLI
stdout/stderr suppressed because they may include private media filenames.
Kaggle datasets create is private by default; NEVER use --public.
Independent metadata check required before claiming remote ACL verification.
"""
from __future__ import annotations
import argparse,hashlib,json,os,re,shutil,subprocess,tempfile,zipfile
from pathlib import Path
from tools.eb01_gpu_watchdog import validate_actual_pilot

KERNEL_RE=re.compile(r"^wolf19loner92/eb01-ltx2b-pilot33-[0-9a-f]{12}$")
HEX64=re.compile(r"^[0-9a-f]{64}$")

class PrivateArchiveBlocked(ValueError): pass

def safe_cmd_result(argv,timeout=180):
    """Run Kaggle quietly and return a bounded failure class, never raw output."""
    if "--public" in argv:
        raise PrivateArchiveBlocked("Never publish original actor video")
    try:
        p=subprocess.run(argv,stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         timeout=timeout,check=False,text=True)
    except subprocess.TimeoutExpired:
        return False,"TIMEOUT"
    except OSError:
        return False,"CLIENT_UNAVAILABLE"
    if p.returncode==0:
        return True,"OK"
    evidence=(p.stdout or "")+" "+(p.stderr or "")
    return False,classify_kaggle_failure(evidence)


def classify_kaggle_failure(evidence):
    """Map Kaggle CLI diagnostics to a safe status without exposing raw text."""
    evidence=str(evidence).lower()
    # Permission/auth failures take precedence over a generic "not found".
    if re.search(r"\b403\b",evidence) or "forbidden" in evidence or "permission denied" in evidence:
        return "ACCESS_DENIED"
    if re.search(r"\b401\b",evidence) or "unauthorized" in evidence:
        return "AUTH_DENIED"
    # Only an explicit HTTP 404 is allowed to authorize first-time creation.
    # Generic missing-file text and transport failures remain UNKNOWN.
    if re.search(r"\b404\b",evidence):
        return "NOT_FOUND"
    if "timed out" in evidence or "connection reset" in evidence:
        return "TIMEOUT"
    return "COMMAND_FAILED"


def safe_cmd(argv,timeout=180):
    return safe_cmd_result(argv,timeout=timeout)[0]

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

def verify_remote_dataset(meta, expected, temp_root):
    """Read back owner ACL and re-download the private dataset to verify exact bytes.

    This downloads the archived Kaggle dataset, NOT the source kernel output.
    It is safe to repeat and makes hourly watchdog runs idempotent.
    """
    metadata_dir=Path(temp_root)/"remote_metadata"
    metadata_dir.mkdir(mode=0o700,parents=True,exist_ok=True)
    metadata_ok,metadata_error=safe_cmd_result(
        ["kaggle","datasets","metadata",meta["id"],"-p",str(metadata_dir)],
        timeout=100)
    if not metadata_ok:
        if metadata_error=="NOT_FOUND":
            return {"exists":False,"status":"REMOTE_DATASET_NOT_FOUND"}
        # 401/403, network errors and unclassified failures are NOT evidence
        # that the dataset is absent. Never create on an ambiguous lookup.
        return {"exists":None,"private":False,"sha_verified":False,
                "status":"REMOTE_DATASET_LOOKUP_"+metadata_error}
    remote_file=metadata_dir/"dataset-metadata.json"
    try:
        remote=json.loads(remote_file.read_text(encoding="utf-8"))
    except (OSError,ValueError,UnicodeError):
        return {"exists":None,"private":False,"sha_verified":False,
                "status":"REMOTE_DATASET_METADATA_UNREADABLE"}
    if remote.get("isPrivate") is not True:
        if remote.get("isPrivate") is False:
            # A visibility conflict is a hard stop, not permission for an
            # automated job to mutate or delete an existing owner dataset.
            # Preserve the evidence and require explicit owner-side resolution.
            return {"exists":True,"private":False,"sha_verified":False,
                    "delete_attempted":False,"status":"REMOTE_DATASET_PRIVACY_CONFLICT"}
        return {"exists":True,"private":False,"sha_verified":False,
                "status":"REMOTE_DATASET_PRIVACY_NOT_PROVEN"}
    download_dir=Path(temp_root)/"remote_dataset_verify"
    download_dir.mkdir(mode=0o700,parents=True,exist_ok=True)
    if not safe_cmd(["kaggle","datasets","download","-d",meta["id"],
                     "-p",str(download_dir),"--force"],timeout=180):
        return {"exists":True,"private":True,"sha_verified":False,
                "status":"REMOTE_PRIVATE_DATASET_DOWNLOAD_UNVERIFIED"}
    archives=list(download_dir.glob("*.zip"))
    if len(archives)!=1:
        return {"exists":True,"private":True,"sha_verified":False,
                "status":"REMOTE_PRIVATE_DATASET_ARCHIVE_MISSING"}
    try:
        with zipfile.ZipFile(archives[0]) as archive_zip:
            names=archive_zip.namelist()
            movie_name=next(n for n in names
                            if Path(n).name=="S02_ORIGINAL_33F_OWNER_REVIEW.mp4")
            receipt_name=next(n for n in names
                              if Path(n).name=="ORIGINAL_RECEIPT_SAFE.json")
            movie_sha=hashlib.sha256(archive_zip.read(movie_name)).hexdigest()
            receipt=json.loads(archive_zip.read(receipt_name).decode("utf-8"))
    except (OSError,KeyError,StopIteration,ValueError,UnicodeError,zipfile.BadZipFile):
        return {"exists":True,"private":True,"sha_verified":False,
                "status":"REMOTE_PRIVATE_DATASET_CONTENTS_INVALID"}
    if movie_sha!=expected or receipt.get("owner_original_mp4_sha256")!=expected:
        return {"exists":True,"private":True,"sha_verified":False,
                "status":"REMOTE_PRIVATE_DATASET_SHA_MISMATCH"}
    return {"exists":True,"private":True,"sha_verified":True,
            "status":"OWNER_PRIVATE_DATASET_ACL_AND_SHA_VERIFIED"}


def _archive_verified_output(kernel,expected,output,temp_parent):
    if not KERNEL_RE.fullmatch(kernel) or not HEX64.fullmatch(expected):
        raise PrivateArchiveBlocked("Kaggle source and SHA not pinned")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise PrivateArchiveBlocked("Owner credential unavailable")
    output=Path(output)
    if output.is_symlink() or not output.is_dir():
        raise PrivateArchiveBlocked("Existing private output directory missing or symlinked")
    if any(node.is_symlink() for node in (output,*output.parents)):
        raise PrivateArchiveBlocked("Symlinked source output ancestor refused")
    video=output/"S02_PILOT33.mp4"
    receipt=output/"EB01_S02_LTX_PILOT33_JOB.json"
    if (video.is_symlink() or receipt.is_symlink() or not video.is_file()
            or not receipt.is_file() or sha(video)!=expected):
        raise PrivateArchiveBlocked("Existing original MP4/receipt/SHA not verified")
    source=validate_actual_pilot(output)
    if (source.get("original_pilot_sha256")!=expected or
        source.get("decoded_true_neural_frames")!=33 or
        source.get("artifact_scan_status")!="NO_NEON_ARTIFACT_FOUND" or
        source.get("full_human_artistic_approval") is not False):
        raise PrivateArchiveBlocked("Original RGB/native/human-review boundary failed")
    original=json.loads(receipt.read_text(encoding="utf-8"))
    if original.get("success") is not True or original.get("clip_sha256")!=expected:
        raise PrivateArchiveBlocked("Original GPU receipt does not match exact SHA")
    safe={k:original.get(k) for k in (
        "schema","shot","success","status","model","mode","frame_count",
        "fps","actual_gpu_seconds","width","height","clip_sha256",
        "true_gpu_neural_inference","human_artistic_approved",
        "whole_movie_ready","published","audio_conditioned_lipsync_verified")}
    if safe.get("whole_movie_ready") is not False:
        raise PrivateArchiveBlocked("A short pilot must never be mislabeled a finished movie")
    # Stable archive receipt field used by remote byte-for-byte readback.
    safe["owner_original_mp4_sha256"]=expected
    safe["watchdog_verification"]={
        "original_pilot_sha256":source["original_pilot_sha256"],
        "decoded_true_neural_frames":source["decoded_true_neural_frames"],
        "artifact_scan_status":source["artifact_scan_status"],
        "narrow_neon_suspected_frames":source.get("narrow_neon_suspected_frames",[]),
        "native_luma_motion_proxy":source.get("native_luma_motion_proxy"),
        "full_human_artistic_approval":False,
        "full_episode_ready":False}
    suffix=kernel.rsplit("-",1)[-1]
    meta=private_dataset_meta(kernel,expected)
    report={"schema":1,"gpu_restarted":False,"published":False,
        "full_movie_ready":False,"public_github_media_upload":False,
        "owner_original_mp4_sha256":expected,"independent_remote_acl":False,
        "remote_original_sha_verified":False}
    # Idempotency: if this exact dataset already exists, verify its privacy
    # metadata and downloaded bytes rather than attempting a duplicate create.
    with tempfile.TemporaryDirectory(prefix="eb01-private-",dir=str(temp_parent)) as d:
        root=Path(d)
        existing=verify_remote_dataset(meta,expected,root/"existing_check")
        if existing.get("exists") is True:
            report["status"]=existing["status"]
            report["independent_remote_acl"]=existing.get("private") is True
            report["remote_original_sha_verified"]=existing.get("sha_verified") is True
            report["owner_private_kaggle_dataset"]=meta["id"]
            report["existing_dataset_reused"]=True
            return report
        if existing.get("status")!="REMOTE_DATASET_NOT_FOUND":
            report["status"]=existing.get("status","REMOTE_DATASET_LOOKUP_UNVERIFIED")
            report["archive_create_blocked"]=True
            return report
        stage=root/"owner_private_dataset"
        stage.mkdir(mode=0o700)
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
        verified=verify_remote_dataset(meta,expected,root/"post_create_check")
        report["independent_remote_acl"]=verified.get("private") is True
        report["remote_original_sha_verified"]=verified.get("sha_verified") is True
        report["status"]=(
            "OWNER_PRIVATE_DATASET_CREATED_ACL_AND_SHA_VERIFIED"
            if verified.get("status")=="OWNER_PRIVATE_DATASET_ACL_AND_SHA_VERIFIED"
            else verified.get("status","OWNER_PRIVATE_DATASET_CREATED_BUT_VERIFICATION_FAILED"))
        report["owner_private_kaggle_dataset"]=meta["id"]
        report["original_video_byte_count"]=video.stat().st_size
        report["source_rgb_scan_status"]=source["artifact_scan_status"]
        report["source_native_luma_motion_proxy"]=source.get("native_luma_motion_proxy")
        return report


def archive_existing_output(kernel,expected,output,temp_parent):
    """Archive the SAME successful watchdog download; never call kernels output again."""
    root=Path(temp_parent).resolve()
    source=Path(output)
    if source.is_symlink() or any(node.is_symlink() for node in (source,*source.parents)):
        raise PrivateArchiveBlocked("Symlinked existing-output path refused")
    resolved=source.resolve()
    if not resolved.is_relative_to(root):
        raise PrivateArchiveBlocked("Existing output must remain under ephemeral RUNNER_TEMP")
    if any((node/".git").exists() for node in (resolved.parent,*resolved.parents)):
        raise PrivateArchiveBlocked("Private source cannot be inside a Git checkout")
    return _archive_verified_output(kernel,expected,resolved,root)


def archive(kernel,expected,temp_parent):
    """Fail closed: never perform a second independent Kaggle kernel-output fetch."""
    if not KERNEL_RE.fullmatch(kernel) or not HEX64.fullmatch(expected):
        raise PrivateArchiveBlocked("Kaggle source and SHA not pinned")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise PrivateArchiveBlocked("Owner credential unavailable")
    raise PrivateArchiveBlocked(
        "Separate Kaggle kernel-output downloads are disabled; archive only the "
        "already verified private_download from the same watchdog run")



def main():
    p=argparse.ArgumentParser()
    p.add_argument("--kernel",required=True)
    p.add_argument("--sha256",required=True)
    p.add_argument("--existing-output-dir",type=Path)
    args=p.parse_args()
    temp_parent=Path(os.environ["RUNNER_TEMP"])
    if args.existing_output_dir:
        report=archive_existing_output(args.kernel,args.sha256,args.existing_output_dir,temp_parent)
    else:
        report=archive(args.kernel,args.sha256,temp_parent)
    out=temp_parent/"EB01_PRIVATE_ARCHIVE_STATUS_ONLY.json"
    out.write_text(json.dumps(report,indent=2)+"\n")
    print("EB01_OWNER_PRIVATE_VIDEO_ARCHIVE",json.dumps(report,sort_keys=True))
    return 0 if report["status"] in (
        "OWNER_PRIVATE_DATASET_CREATED_ACL_AND_SHA_VERIFIED",
        "OWNER_PRIVATE_DATASET_ACL_AND_SHA_VERIFIED") else 2

if __name__=="__main__":
    raise SystemExit(main())
