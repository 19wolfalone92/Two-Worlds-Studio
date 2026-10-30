"""Private Kaggle shorts worker: bounded, resumable, no publication or fabricated media.

A Kaggle kernel (or a local Python process) calls `run_job` with a trusted
configuration. All private source materials are mounted as private Kaggle datasets.
GPU generation is opt-in and defaults to CPU-only preflight; no credentials ship.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable

from .offline_media import (accelerator_status, real_image, real_voice, tone_score,
                            release_gpu_models, digest)
from .autonomous_factory import atomic_json, inventory, mood_of
from .shorts_factory import run as render_story

STORIES = {str(i) for i in range(13, 19)}
PHASES = {"preflight", "image", "voice", "music", "render"}
MAX_ARCHIVE = 180 * 1024 * 1024
MAX_EXPANDED = 800 * 1024 * 1024
ALLOWED_PROJECT_PATHS = {"project.json", "story.txt", "shot_plan.json", "local_factory_receipt.json"}


class KaggleBlocked(RuntimeError):
    pass


def _safe_extract(z: zipfile.ZipFile, target: Path) -> None:
    files = z.infolist()
    if len(files) > 500:
        raise KaggleBlocked("Input archive contains too many files")
    total = 0
    root = target.resolve()
    for item in files:
        pure = Path(item.filename)
        if (pure.is_absolute() or any(p in ("..", "") for p in pure.parts)
                or re.match(r"^[A-Za-z]:", item.filename)):
            raise KaggleBlocked("Unsafe ZIP path")
        total += item.file_size
        if total > MAX_EXPANDED or item.file_size > MAX_ARCHIVE:
            raise KaggleBlocked("Archive expanded size exceeds safety limit")
        # Refuse symlinks and hardlinks encoded via Unix external attrs.
        if (item.external_attr >> 16) & 0o170000 == 0o120000:
            raise KaggleBlocked("Input ZIP contains a symlink")
        result = (target / pure).resolve()
        if root not in result.parents and result != root:
            raise KaggleBlocked("ZIP would escape target folder")
    z.extractall(target)


def _locate_project(root: Path, story: str) -> Path:
    paths = [p for p in root.rglob("project.json")
             if p.parent.name == story and p.parent.parent.name == "production_queue"]
    if len(paths) != 1:
        raise KaggleBlocked(f"Expected one PRIVATE production_queue/{story}/project.json, found {len(paths)}")
    return paths[0].parent


def prepare_input(private_source: Path, target: Path, story: str, *, recovery: Path | None = None) -> Path:
    """Open private folder or ZIP into new writable work dir, preserving original SHA.

    Only a single original story is copied, to limit output and GPU session I/O.
    Existing scene assets can be supplied by an attached private recovery archive.
    """
    if story not in STORIES:
        raise KaggleBlocked("Only original stories 13..18 are permitted")
    private_source = Path(private_source).resolve()
    target = Path(target).resolve()
    if target.exists():
        raise KaggleBlocked("Work folder exists; refusing overwrites")
    staging = target.parent / (".staging_" + story)
    if staging.exists():
        raise KaggleBlocked("Staging folder exists")
    staging.mkdir(parents=True)
    try:
        if private_source.is_file() and private_source.suffix.lower() == ".zip":
            if private_source.stat().st_size > MAX_ARCHIVE:
                raise KaggleBlocked("Private source ZIP too large")
            with zipfile.ZipFile(private_source) as z:
                _safe_extract(z, staging)
            original = _locate_project(staging, story)
        elif private_source.is_dir():
            original = _locate_project(private_source, story)
        else:
            raise KaggleBlocked("Private story input ZIP/directory not found")
        # Copy only one story, preserving the locked SHA and associated assets.
        root = staging / "chosen"
        if root.exists():
            raise KaggleBlocked("Chosen folder conflicts with input archive")
        shutil.copytree(original, root, symlinks=False)
        if recovery:
            _import_recovery(Path(recovery), root, story)
        state = inventory(root)  # hashes and source story preserved.
        if state["id"] != f"work_{story}":
            raise KaggleBlocked("Source story does not match expected story ID")
        os.replace(root, target)
        return target
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _import_recovery(recovery: Path, root: Path, story: str) -> None:
    recovery = recovery.resolve()
    if not recovery.is_file() or recovery.suffix.lower() != ".zip":
        raise KaggleBlocked("Recovery must be a private ZIP from a previous run")
    with tempfile.TemporaryDirectory(prefix=".recovery-") as tmp:
        temp = Path(tmp)
        with zipfile.ZipFile(recovery) as z:
            _safe_extract(z, temp)
        directories = [p for p in temp.rglob("project.json") if p.parent.name == story]
        if len(directories) != 1:
            raise KaggleBlocked("Recovery project missing or ambiguous")
        previous = directories[0].parent
        audit_path = temp / "run_manifest.json"
        if not audit_path.is_file():
            raise KaggleBlocked("Missing SHA-256 recovery manifest")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("story") != story or not isinstance(audit.get("generated_media"), list):
            raise KaggleBlocked("Recovery manifest belongs to another story")
        for entry in audit["generated_media"]:
            rel = Path(entry.get("path", ""))
            if rel.is_absolute() or ".." in rel.parts:
                raise KaggleBlocked("Unsafe recovery inventory")
            file = (temp / rel).resolve()
            if (temp.resolve() not in file.parents or not file.is_file() or
                    digest(file) != entry.get("sha256")):
                raise KaggleBlocked("Recovery ZIP content differs from SHA-256 manifest")
        prior_doc = json.loads((previous / "project.json").read_text(encoding="utf-8"))
        doc = json.loads((root / "project.json").read_text(encoding="utf-8"))
        if audit.get("script_sha256") != doc.get("script_sha256"):
            raise KaggleBlocked("Recovery story fingerprint does not match locked original")
        if (prior_doc.get("script_sha256") != doc.get("script_sha256")
                or prior_doc.get("id") != doc.get("id")):
            raise KaggleBlocked("Recovery belongs to another story or script revision")
        by_id = {scene["id"]: scene for scene in prior_doc["scenes"]}
        if set(by_id) != {scene["id"] for scene in doc["scenes"]}:
            raise KaggleBlocked("Recovery scene IDs differ")
        for curr in doc["scenes"]:
            old = by_id[curr["id"]]
            for key, extension in (("image", ".png"), ("voice", ".wav"),
                                   ("music", ".wav")):
                rel = old.get(key)
                if not rel:
                    continue
                pure = Path(rel)
                if pure.is_absolute() or ".." in pure.parts or pure.suffix.lower() != extension:
                    raise KaggleBlocked("Invalid recovery asset filename")
                source = (previous / pure).resolve()
                if previous.resolve() not in source.parents or not source.is_file():
                    continue
                dest = (root / pure).resolve()
                if root.resolve() not in dest.parents:
                    raise KaggleBlocked("Recovery asset escapes project")
                if dest.exists():
                    if digest(dest) != digest(source):
                        raise KaggleBlocked("Different asset already exists; refuse overwrite")
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
                curr[key] = rel
            if old.get("subtitles") and (root / old["voice"]).exists():
                curr["subtitles"] = old["subtitles"]
        atomic_json(root / "project.json", doc)
        inventory(root)


def _check_model_home(root: Path, phase: str) -> Path:
    """No silent model downloads; model stage must be explicitly provided."""
    m = Path(root).resolve() / ("image" if phase == "image" else "voice")
    if not m.is_dir():
        raise KaggleBlocked(f"Required mounted/offline model directory missing: {m}")
    # Diffusers pipelines usually use model_index.json; Qwen uses config.json.
    accepted = ("model_index.json", "config.json") if phase == "image" else ("config.json",)
    if not any((m / name).is_file() for name in accepted):
        raise KaggleBlocked(f"Model weights / config missing from {m}")
    return m


def _gpu_info() -> dict:
    info = accelerator_status()
    try:
        import torch
        info["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(info["device_count"])]
        info["free_gb"] = [round(torch.cuda.mem_get_info(i)[0] / 1024**3, 2)
                           for i in range(info["device_count"])]
    except (ImportError, RuntimeError):
        info["device_count"] = 0
    return info


def run_job(workdir: Path, out: Path, model_root: Path, *, story: str, phase: str,
            first_scene: int = 1, max_scenes: int = 2, gpu_id: int = 0,
            time_budget_s: int = 2400, image_size: tuple[int, int] = (576, 1024),
            speaker: str = "Serena", image_backend: Callable | None = None,
            voice_backend: Callable | None = None, clock: Callable = time.monotonic) -> dict:
    """One bounded Kaggle session. Everything written to disk before returning.

    A mocked backend is used ONLY in CPU tests. The real backend requires local
    weights and an active CUDA device. Results always remain unreviewed.
    """
    if phase not in PHASES or story not in STORIES:
        raise KaggleBlocked("Invalid story/phase")
    if not (1 <= first_scene <= 8 and 1 <= max_scenes <= 8):
        raise KaggleBlocked("Bounded scene range must be 1..8")
    if not 60 <= time_budget_s <= 18000:
        raise KaggleBlocked("Run budget must be 60..18000 seconds")
    if gpu_id not in (0, 1):
        raise KaggleBlocked("Choose GPU 0 or 1")
    if image_size[0] % 16 or image_size[1] % 16:
        raise KaggleBlocked("Image size must be multiples of 16")
    out = Path(out).resolve()
    root = Path(workdir).resolve()
    if not root.is_dir() or not (root / "project.json").is_file():
        raise KaggleBlocked("Private working story was not prepared")
    state = inventory(root)
    if state["id"] != f"work_{story}":
        raise KaggleBlocked("Story mismatch")
    if out.exists() and any(out.iterdir()):
        raise KaggleBlocked("Output dir must be new/empty")
    out.mkdir(parents=True, exist_ok=True)
    started = clock()
    status = {"schema": 1, "story": story, "phase": phase, "status": "IN_PROGRESS",
              "gpu": _gpu_info(), "made": [], "errors": [], "skipped": [],
              "time_budget_s": time_budget_s, "publication_approved": False,
              "used_paid_services": False}
    scenes = json.loads((root / "project.json").read_text(encoding="utf-8"))["scenes"]
    shots = json.loads((root / "shot_plan.json").read_text(encoding="utf-8"))
    by_id = {s["scene"]: s for s in shots}
    selected = [s for s in scenes if first_scene <= s["id"] < first_scene + max_scenes]
    if not selected:
        raise KaggleBlocked("No scenes selected")

    def checkpoint():
        status["source_sha256"] = json.loads((root/"project.json").read_text(encoding="utf-8"))["script_sha256"]
        status["elapsed_s"] = round(clock() - started, 2)
        atomic_json(out / "run_status.json", status)
    checkpoint()
    if phase == "preflight":
        status["status"] = "PREFLIGHT_OK"
        status["scene_count"] = len(scenes)
    elif phase in ("image", "voice"):
        if image_backend is None and voice_backend is None:
            if status["gpu"].get("device_count", 0) <= gpu_id:
                raise KaggleBlocked("GPU not available; choose T4 x2 accelerator before starting")
            _check_model_home(model_root, phase)
        if phase == "image":
            backend = image_backend or real_image
            model = None
            if backend is real_image:
                from diffusers import Flux2KleinPipeline
                import torch
                model_path = _check_model_home(model_root, "image")
                # Only CUDA visible device selected (set before torch import on kernel).
                model = Flux2KleinPipeline.from_pretrained(
                    str(model_path), torch_dtype=torch.float16,
                    local_files_only=True)
                model.enable_model_cpu_offload()
            try:
                for scene in selected:
                    if clock() - started > time_budget_s:
                        status["status"] = "TIME_BUDGET_REACHED"
                        break
                    target = root / scene["image"]
                    if target.exists():
                        status["skipped"].append(scene["id"]); continue
                    shot = by_id[scene["id"]]
                    seed = int(hashlib.sha256((f"work_{story}:{scene['id']}").encode()).hexdigest()[:8],16)
                    try:
                        args = dict(seed=seed, width=image_size[0], height=image_size[1])
                        if model is not None:
                            args["pipe"] = model
                        result = backend(shot["visual_prompt_ru"], target, model_root, **args)
                        if not target.is_file():
                            raise KaggleBlocked("Image generator reported success without a real file")
                        status["made"].append({"scene": scene["id"], "image_sha256": digest(target),
                                                "model": result.get("model"),
                                                "native_resolution": [image_size[0], image_size[1]],
                                                "artistic_review": "pending", "identity_verified": False})
                    except Exception as exc:
                        status["errors"].append({"scene": scene["id"], "type": type(exc).__name__,
                                                  "message": str(exc)[:350]})
                    checkpoint()
            finally:
                del model
                release_gpu_models()
        else:
            backend = voice_backend or real_voice
            try:
                for scene in selected:
                    if clock() - started > time_budget_s:
                        status["status"] = "TIME_BUDGET_REACHED"
                        break
                    target = root / scene["voice"]
                    if target.exists() and scene["subtitles"]:
                        status["skipped"].append(scene["id"]); continue
                    if target.exists() and not scene["subtitles"]:
                        status["errors"].append({"scene": scene["id"], "message":
                                                  "Existing voice lacks aligned subtitles; refuse overwrite"})
                        continue
                    shot = by_id[scene["id"]]
                    try:
                        result = backend(shot["exact_source_text"], target, model_root,
                                         speaker=speaker, style=mood_of(shot),device="cuda")
                        if not target.is_file() or not result.get("subtitles"):
                            raise KaggleBlocked("Voice missing real audio/timestamps")
                        doc = json.loads((root / "project.json").read_text(encoding="utf-8"))
                        match = next(s for s in doc["scenes"] if s["id"] == scene["id"])
                        match["subtitles"] = result["subtitles"]
                        atomic_json(root / "project.json", doc)
                        status["made"].append({"scene": scene["id"], "voice_sha256": digest(target),
                                                "seconds": result.get("seconds")})
                    except Exception as exc:
                        status["errors"].append({"scene": scene["id"], "type": type(exc).__name__,
                                                  "message": str(exc)[:350]})
                    checkpoint()
            finally:
                release_gpu_models()
    elif phase == "music":
        for scene in selected:
            if clock()-started > time_budget_s:
                status["status"] = "TIME_BUDGET_REACHED"; break
            voice_path = root / scene["voice"]
            if not voice_path.is_file():
                status["errors"].append({"scene":scene["id"], "message":"Real voice not present"})
                continue
            target = root / f"music/scene_{scene['id']:02d}.wav"
            if target.exists():
                status["skipped"].append(scene["id"])
                continue
            try:
                from .shorts_factory import duration
                tone_score(target, duration(voice_path), mood=mood_of(by_id[scene["id"]]),
                           seed=scene["id"])
                doc=json.loads((root/"project.json").read_text(encoding="utf-8"))
                next(s for s in doc["scenes"] if s["id"] == scene["id"])["music"] = f"music/scene_{scene['id']:02d}.wav"
                atomic_json(root / "project.json", doc)
                status["made"].append({"scene":scene["id"],"music_sha256":digest(target)})
            except Exception as exc:
                status["errors"].append({"scene":scene["id"],"message":str(exc)[:350]})
            checkpoint()
    elif phase == "render":
        try:
            from .shorts_factory import validate_project
            validate_project(root / "project.json")
            result=render_story(root / "project.json", out)
            status["render"]=result
        except Exception as exc:
            status["errors"].append({"component":"render","message":str(exc)[:400]})
    if status["status"] == "IN_PROGRESS":
        status["status"] = "PARTIAL" if status["errors"] else "COMPLETE"
    status["inventory"] = inventory(root)["missing"]
    checkpoint()
    return status


def export_private_assets(workdir: Path, out_zip: Path, run_status: dict) -> dict:
    """Create a private resumable ZIP; never mark it approved for publication."""
    workdir = Path(workdir).resolve()
    doc = json.loads((workdir / "project.json").read_text(encoding="utf-8"))
    story = str(doc["id"]).replace("work_", "")
    if story not in STORIES:
        raise KaggleBlocked("Invalid work story")
    from .autonomous_factory import inventory
    inventory(workdir)
    out_zip = Path(out_zip)
    if out_zip.exists():
        raise KaggleBlocked("Output archive already exists")
    out_zip.parent.mkdir(parents=True,exist_ok=True)
    manifest = {"schema":1,"status":run_status.get("status"),
                "story":story,"script_sha256":doc["script_sha256"],
                "generated_media":[],"reviewed":False,"published":False}
    project_files = [p for p in workdir.rglob("*") if p.is_file()]
    with zipfile.ZipFile(out_zip,"x",zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in project_files:
            rel = f.relative_to(workdir)
            if rel.parts[0] not in {"images","voices","music"} and rel.as_posix() not in ALLOWED_PROJECT_PATHS:
                continue
            if f.stat().st_size > MAX_ARCHIVE:
                raise KaggleBlocked("Single asset too large for recovery ZIP")
            arc = Path("production_queue") / story / rel
            z.write(f,str(arc))
            manifest["generated_media"].append({"path":str(arc),"sha256":digest(f),
                                                "bytes":f.stat().st_size})
        z.writestr("run_manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2))
    return {"zip":str(out_zip),"sha256":digest(out_zip),"files":len(manifest["generated_media"]),
            "reviewed":False,"published":False}


def main(argv=None):
    p=argparse.ArgumentParser(description="Private Kaggle production worker (no auto publication)")
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--recovery",type=Path)
    p.add_argument("--models",type=Path,default=Path("/kaggle/input/two-worlds-models"))
    p.add_argument("--work",type=Path,default=Path("/kaggle/working/two_worlds_project"))
    p.add_argument("--out",type=Path,default=Path("/kaggle/working"))
    p.add_argument("--story",choices=sorted(STORIES),required=True)
    p.add_argument("--phase",choices=sorted(PHASES),default="preflight")
    p.add_argument("--start",type=int,default=1)
    p.add_argument("--count",type=int,default=2)
    p.add_argument("--gpu-id",type=int,choices=(0,1),default=0)
    p.add_argument("--time-budget",type=int,default=2400)
    p.add_argument("--speaker",default="Serena")
    a=p.parse_args(argv)
    # Restrict visible GPUs before importing torch/diffusers; T4x2 provides 16+16GB,
    # *not* a 32GB unified allocation.
    if a.phase in ("image","voice"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu_id)
    output=a.out.resolve()
    output.mkdir(parents=True,exist_ok=True)
    receipt_path=output/"two_worlds_kaggle_receipt.json"
    prepared=False
    try:
        prepare_input(a.source,a.work,a.story,recovery=a.recovery)
        prepared=True
        result=run_job(a.work,output/"render",a.models,story=a.story,
                       phase=a.phase,first_scene=a.start,max_scenes=a.count,
                       time_budget_s=a.time_budget,speaker=a.speaker,gpu_id=0)
        incremental_ok = (result["status"] in {"TIME_BUDGET_REACHED", "PARTIAL"} and
                          len(result.get("made", [])) > 0)
        receipt={"success":result["status"] in {"COMPLETE","PREFLIGHT_OK"} or incremental_ok,
                 "phase_complete":result["status"] in {"COMPLETE","PREFLIGHT_OK"},
                 "result":result,"reviewed":False,"published":False}
    except Exception as exc:
        receipt={"success":False,"error_type":type(exc).__name__,
                 "message":str(exc)[:900],"reviewed":False,"published":False}
    try:
        if prepared:
            receipt["asset_bundle"]=export_private_assets(a.work,
                output/f"TwoWorlds_PRIVATE_{a.story}_{a.phase}.zip",receipt.get("result",{}))
    except Exception as exc:
        receipt["bundle_error"]={"type":type(exc).__name__,"message":str(exc)[:500]}
        receipt["success"]=False
    atomic_json(receipt_path,receipt)
    print(json.dumps({"success":receipt["success"],"receipt":str(receipt_path),
                      "bundle":receipt.get("asset_bundle",{}).get("zip"),
                      "error":receipt.get("message")},ensure_ascii=False))
    return 0 if receipt["success"] else 2


if __name__=="__main__":
    raise SystemExit(main())