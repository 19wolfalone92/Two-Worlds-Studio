"""Explicit, legally gated installer for official local neural video model weights.

Weights live *only* in the operator's ComfyUI model directory, never in the
public TWO WORLDS Git repository. Default is offline inventory; the download
flag performs a real HF transfer after separate licence acknowledgement.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
from pathlib import Path

# (official HF repository, remote repo-relative path, local ComfyUI model subdir)
COMPONENTS = {
    "wan22-5b": [
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
         "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", "diffusion_models"),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
         "split_files/vae/wan2.2_vae.safetensors", "vae"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
         "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    ],
    "ltx25-dfr": [
        ("Lightricks/LTX-2.5",
         "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", "diffusion_models"),
        ("Lightricks/LTX-2.5",
         "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", "text_encoders"),
        ("Lightricks/LTX-2.5",
         "vae/ltx-2.5-video-vae-bf16.safetensors", "vae"),
        ("Lightricks/LTX-2.5",
         "vae/ltx-2.5-audio-vae-bf16.safetensors", "vae"),
        ("Lightricks/LTX-2.5",
         "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
         "latent_upscale_models"),
        ("Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler",
         "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors", "loras"),
    ],
}
LICENSES = {
    "wan22-5b": "Wan2.2 / Apache-2.0; independently check model-specific weights terms",
    "ltx25-dfr": "LTX-2.x Community License; accept HF gated terms and verify commercial eligibility",
}


def _safe_root(comfy: Path) -> Path:
    root = Path(comfy).expanduser().resolve()
    if not (root / "models").is_dir():
        raise ValueError("Supply an installed private ComfyUI directory containing models/")
    if root == Path.cwd().resolve():
        # Avoid staging gigabytes beside the public Git sources accidentally.
        raise ValueError("The current repository cannot be used as a model installation root")
    return root


def inventory(model: str, root: Path) -> dict:
    if model not in COMPONENTS:
        raise ValueError("Unknown reviewed checkpoint family")
    root = _safe_root(root)
    files = []
    for repository, path, kind in COMPONENTS[model]:
        target = root / "models" / kind / Path(path).name
        if target.is_symlink():
            raise ValueError("Model target is a symlink; refuse unsafe checkpoint storage")
        files.append({"repo": repository, "remote_path": path, "target": str(target),
                      "installed": target.is_file(), "size_bytes": target.stat().st_size if target.is_file() else None})
    return {"model": model, "license": LICENSES[model], "files": files,
            "download_performed": False, "gpu_inference": False, "published": False}


def install(model: str, root: Path, *, accept_license: bool, execute: bool) -> dict:
    report = inventory(model, root)
    if not execute:
        return {"status": "OFFLINE_INVENTORY", **report}
    if not accept_license:
        raise PermissionError("Accept official weight licence before staging real checkpoint files")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install huggingface_hub>=0.29") from exc

    root = _safe_root(root)
    api = HfApi()
    revisions = {}
    per_repo = {}
    for entry in report["files"]:
        repo, remote = entry["repo"], entry["remote_path"]
        if repo not in per_repo:
            meta = api.model_info(repo_id=repo, files_metadata=True)
            if not isinstance(meta.sha, str) or len(meta.sha) != 40:
                raise RuntimeError("Hugging Face did not supply a fixed SHA revision")
            per_repo[repo] = {f.rfilename: f for f in meta.siblings}
            revisions[repo] = meta.sha
        obj = per_repo[repo].get(remote)
        if obj is None or type(obj.size) is not int or not 1024 <= obj.size <= 80 * 1024**3:
            raise RuntimeError("Unexpected/missing upstream checkpoint; stop instead of guessing")
        entry["expected_bytes"] = obj.size
        # hf_hub_download validates HTTP integrity at the pinned repository revision.
        # Check published LFS SHA-256 again after staging when metadata supplies it.
        lfs = getattr(obj, "lfs", None)
        entry["expected_sha256"] = lfs.get("sha256") if isinstance(lfs, dict) else None
    missing = [e for e in report["files"] if not e["installed"]]
    total = sum(e["expected_bytes"] for e in missing)
    if shutil.disk_usage(root).free < total + 2 * 1024**3:
        raise OSError(errno.ENOSPC, "Not enough private disk space for this model set")

    cache = root / "models" / ".two_worlds_hf_cache"
    staged = []
    for entry in report["files"]:
        target = Path(entry["target"])
        if target.is_file():
            if target.stat().st_size != entry["expected_bytes"]:
                raise RuntimeError("Existing model file has the wrong size; inspect it manually")
            continue
        repo = entry["repo"]
        local = cache / repo.replace("/", "__")
        raw = Path(hf_hub_download(
            repo_id=repo, filename=entry["remote_path"], revision=revisions[repo],
            local_dir=local,
        ))
        if raw.stat().st_size != entry["expected_bytes"]:
            raise RuntimeError("Downloaded model file size does not match pinned HF metadata")
        if entry["expected_sha256"]:
            digest = hashlib.sha256()
            with raw.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["expected_sha256"]:
                raise RuntimeError("Downloaded model file has a mismatched official LFS SHA-256")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Stage as hardlinks, avoiding a second 10-66GiB copy on the same disk.
        # No symlink to GitHub repo, no automatic archive upload.
        os.link(raw, target)
        staged.append(str(target))
    return {"status": "WEIGHTS_STAGED_PRIVATELY_GPU_NOT_RUN", "model": model,
            "pinned_revisions": revisions, "installed_files": staged,
            "license": LICENSES[model], "download_performed": bool(staged),
            "gpu_inference": False, "published": False}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage official legal checkpoints outside the public repository")
    p.add_argument("--model", choices=tuple(COMPONENTS), required=True)
    p.add_argument("--comfy-root", type=Path, required=True)
    p.add_argument("--execute-download", action="store_true")
    p.add_argument("--accept-model-license", action="store_true")
    args = p.parse_args(argv)
    try:
        result = install(args.model, args.comfy_root,
                         accept_license=args.accept_model_license, execute=args.execute_download)
    except (OSError, ValueError, RuntimeError, PermissionError) as exc:
        result = {"status": "BLOCKED", "reason": str(exc),
                  "download_performed": False, "gpu_inference": False, "published": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
