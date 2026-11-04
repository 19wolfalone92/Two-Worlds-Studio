"""CPU-only private Batch03 render relay. No Kaggle GPU, model calls, or posting.

Downloads previously generated PRIVATE source/image/voice archives via the
existing Kaggle authorization, reconciles their SHA-256 manifests, invokes the
real FFmpeg renderer for complete stories, and returns only encrypted output.
No private filenames, narration, token or media are printed in public Actions.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from studio.private_batch_reconcile import reconcile, digest
from studio.autonomous_batch import execute
from tools.batch03_production_relay import seal_return, TEMP as SEAL_TEMP
from tools.batch03_inventory_public import list_owned_kernels, discover_output_kinds

ROOT = Path(__file__).resolve().parents[1]
OWNER = "wolf19loner92"
DATASET = "two-worlds-private-batch03-20261104-repair-v6"
SOURCE_SHA = "0bf78ddd15984ecd51287dc3140cd9b2b89192348fd95fd6e49edf4e95d0f061"
STORIES = tuple(str(i) for i in range(13, 19))
PART_PATTERN = re.compile(r"^TwoWorlds_PRIVATE_(1[3-8])_(image|voice)[.]zip$")
NONCE = re.compile(r"^[a-z0-9-]{6,64}$")


class ReleaseBlocked(RuntimeError):
    pass


def command(*args: str, timeout: int = 180) -> str:
    try:
        completed = subprocess.run(["kaggle", *args], capture_output=True,
                                   text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseBlocked("Private archive read failed: " + type(exc).__name__) from None
    # Raw provider responses may contain unpublished story titles and paths.
    if completed.returncode:
        raise ReleaseBlocked("Private Kaggle download/list operation failed")
    return completed.stdout


def index_private_outputs() -> dict[tuple[str, str], str]:
    kernels, ok = list_owned_kernels()
    if not ok:
        raise ReleaseBlocked("Read-only private Kaggle account inventory failed")
    index = {}
    for handle in kernels:
        try:
            rows = list(csv.DictReader(io.StringIO(
                command("kernels", "files", handle, "--csv", "--page-size", "200",
                        timeout=75))))
        except (ReleaseBlocked, csv.Error):
            continue
        # First matching owner kernel (newest per Kaggle list) wins. No GPU is
        # started, and no missing assets are hallucinated into the release.
        for story, phase in discover_output_kinds(rows):
            if phase in ("image", "voice"):
                index.setdefault((story, phase), handle)
        if all((story, phase) in index for story in STORIES for phase in ("image", "voice")):
            break
    return index


def safe_package(input_root: Path, dest: Path) -> None:
    """Final ZIP contains actual finished review files, never raw source."""
    with zipfile.ZipFile(dest, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for story in STORIES:
            folder = input_root / f"work_{story}"
            if not folder.is_dir():
                continue
            for name in ("final.mp4", "cover.jpg", "subtitles.srt", "qa_report.json", "receipt.json"):
                p = folder / name
                if not p.is_file() or p.is_symlink():
                    raise ReleaseBlocked("Incomplete reviewed render package")
                z.write(p, f"review/work_{story}/{name}")


def source_location(folder: Path) -> Path:
    candidates = list(folder.rglob("batch03_source.zip"))
    if len(candidates) != 1 or candidates[0].is_symlink() or candidates[0].stat().st_size > 180 * 1024**2:
        raise ReleaseBlocked("Exact private original story ZIP missing or ambiguous")
    if digest(candidates[0]) != SOURCE_SHA:
        raise ReleaseBlocked("Private source SHA-256 changed since original authorization")
    return candidates[0]


def release(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise ReleaseBlocked("Encrypted return destination must be new")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("schema") != 1 or config.get("mode") != "cpu_only_review"
            or config.get("free_resources_only") is not True
            or config.get("source_sha256") != SOURCE_SHA
            or config.get("dataset_slug") != DATASET
            or not isinstance(config.get("nonce"), str)
            or not NONCE.fullmatch(config["nonce"])):
        raise ReleaseBlocked("Unapproved or unbounded private CPU batch request")
    if os.environ.get("KAGGLE_USERNAME") != OWNER or not os.environ.get("KAGGLE_API_TOKEN"):
        raise ReleaseBlocked("Existing Kaggle authorization is unavailable")
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema": 1, "nonce": config["nonce"], "mode": "cpu_only_review",
               "gpu_requested": False, "new_gpu_jobs": 0, "paid": False,
               "published": False, "zip_archive_stories": 0, "video_files": 0,
               "status": "INIT", "errors": [], "encrypted_return": False}
    with tempfile.TemporaryDirectory(prefix="tw-private-release-", dir=os.environ.get("RUNNER_TEMP", "/tmp")) as t:
        root = Path(t)
        try:
            # Reuse the already-authorized PRIVATE Kaggle source dataset.
            input_dir = root / "original"; input_dir.mkdir()
            command("datasets", "download", f"{OWNER}/{DATASET}",
                    "--file", "batch03_source.zip", "--path", str(input_dir),
                    "--quiet", timeout=600)
            # Some Kaggle API versions wrap a requested ZIP in another ZIP.
            # Unwrap only that single filename, never extract private content to logs.
            if not list(input_dir.rglob("batch03_source.zip")):
                wrappers = list(input_dir.rglob("batch03_source.zip.zip"))
                if len(wrappers) != 1:
                    raise ReleaseBlocked("The private original source archive is unavailable")
                with zipfile.ZipFile(wrappers[0]) as packaged:
                    candidates = [item for item in packaged.infolist()
                                  if Path(item.filename).name == "batch03_source.zip"
                                  and not Path(item.filename).is_absolute()
                                  and ".." not in Path(item.filename).parts]
                    if len(candidates) != 1:
                        raise ReleaseBlocked("Ambiguous wrapped original source archive")
                    target = input_dir / "batch03_source.zip"
                    with packaged.open(candidates[0]) as incoming, target.open("xb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            source = source_location(input_dir)
            index = index_private_outputs()
            # Never download gigantic motion clips in this free CPU pass:
            # use the tested subpixel camera over the original movie stills.
            media = root / "media"; media.mkdir()
            not_found = []
            for story in STORIES:
                for phase in ("image", "voice"):
                    handle = index.get((story, phase))
                    if not handle:
                        not_found.append((story, phase))
                        continue
                    dst = root / "incoming" / story / phase
                    dst.mkdir(parents=True, exist_ok=True)
                    name = f"TwoWorlds_PRIVATE_{story}_{phase}.zip"
                    # Regex anchored to a trusted exact output basename.
                    command("kernels", "output", handle, "--path", str(dst),
                            "--file-pattern", re.escape(name) + "$",
                            "--force", "--quiet", timeout=600)
                    candidates = list(dst.rglob(name))
                    if len(candidates) != 1 or candidates[0].is_symlink():
                        raise ReleaseBlocked("Expected exact private media ZIP was not downloaded")
                    if candidates[0].stat().st_size > 750 * 1024**2:
                        raise ReleaseBlocked("Private media archive size exceeds bounded limit")
                    shutil.copy2(candidates[0], media / name)
            # Incomplete inventory is still reported; never substitute media.
            summary["zip_archive_stories"] = len({
                story for story in STORIES if all((media / f"TwoWorlds_PRIVATE_{story}_{phase}.zip").is_file()
                                                 for phase in ("image", "voice"))})
            summary["missing_media_zip_count"] = len(not_found)
            joined = reconcile(source, media, root / "queue")
            counts = joined["production"]["counts"]
            summary["queue_status"] = counts
            if counts["READY_TO_RENDER"]:
                result = execute(root / "queue", root / "unpublished_exports", max_jobs=6)
                summary["render_attempts"] = [{k: row[k] for k in ("story", "status")}
                                               for row in result.get("render_attempts", [])]
                summary["video_files"] = result["technical_exports_ready_for_review"]
                if summary["video_files"]:
                    saved = root / "private_reviews.zip"
                    safe_package(root / "unpublished_exports", saved)
                    SEAL_TEMP.mkdir(parents=True, exist_ok=True)
                    seal_return(saved, ROOT / ".github/batch03_requests/recipient_public.pem",
                                output, config["nonce"])
                    summary["encrypted_return"] = True
            summary["status"] = ("SIX_VIDEOS_FOR_REVIEW" if summary["video_files"] == 6
                                 else "PARTIAL_OR_SOURCE_BLOCKED")
        except Exception as exc:
            # No private file paths or provider text in public Action logs.
            summary["status"] = "BLOCKED"
            summary["errors"].append(type(exc).__name__)
        # Even on failure deliver an authenticated, encrypted detailed report
        # only if no sealed MP4 return was created.
        if not summary["encrypted_return"]:
            detail = root / "private_release_status.json"
            detail.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            SEAL_TEMP.mkdir(parents=True, exist_ok=True)
            seal_return(detail, ROOT / ".github/batch03_requests/recipient_public.pem",
                        output, config["nonce"])
            summary["encrypted_return"] = True
    # Note: this JSON contains NO narration, private media names or kernel IDs.
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Reuse PRIVATE Kaggle media for CPU-only six-video review export")
    p.add_argument("--request", type=Path, default=ROOT / ".github/batch03_requests/release.json")
    p.add_argument("--output", type=Path, default=ROOT / "dist/private_batch03_release")
    a = p.parse_args(argv)
    result = release(a.request, a.output)
    return 0 if result["status"] == "SIX_VIDEOS_FOR_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
