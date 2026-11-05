"""One-command private Batch03 runner. No app, account, API or internet required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .autonomous_factory import inventory, run_story, atomic_json


def batch_status(queue_root: Path) -> dict:
    folders = sorted(x for x in Path(queue_root).iterdir()
                     if x.is_dir() and (x / "project.json").is_file())
    if not folders:
        raise ValueError("No prepared Work stories; first run shorts_batch")
    results = [inventory(folder) for folder in folders]
    return {"stories": len(results), "total_scenes": sum(len(s["scenes"]) for s in results),
            "ready_for_video": sum(s["ready_for_video"] for s in results),
            "jobs": results}


def run_batch(queue_root: Path, out: Path, models: Path, *,
              images: bool = False, voices: bool = False,
              music: bool = True, render: bool = True, device: str = "cuda",
              speaker: str = "Serena") -> dict:
    queue_root = Path(queue_root).resolve()
    out = Path(out).resolve()
    initial = batch_status(queue_root)
    outcomes = []
    # Run sequentially to avoid concurrent GPU OOM and model duplicate loading.
    for job in initial["jobs"]:
        outcome = run_story(Path(job["project_dir"]), out, models,
                            images=images, voices=voices, music=music,
                            render=render, device=device, speaker=speaker,
                            force_cpu_voice=(device == "cpu"))
        outcomes.append({"id": job["id"], "status": outcome["status"],
                         "created": outcome["created"], "missing": outcome["missing"],
                         "errors": outcome["errors"]})
    status = {"schema":1, "local_only":True, "charged_credits": 0,
              "total":len(outcomes),
              "completed_for_review":sum(x["status"]=="REVIEW_REQUIRED" for x in outcomes),
              "stories":outcomes, "publishing_allowed":False}
    atomic_json(queue_root / "offline_batch_receipt.json", status)
    return status


def package_review(queue_root: Path, out: Path, zip_path: Path) -> dict:
    """Package complete private review deliveries after verifying every receipt."""
    from .shorts_review import write_review_archive
    jobs = batch_status(queue_root)["jobs"]
    out = Path(out)
    folders = []
    for job in jobs:
        folder = out / job["id"]
        receipt_path = folder / "receipt.json"
        if not receipt_path.is_file():
            raise ValueError(f"Missing finished reviewed-file bundle: {job['id']}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        project = json.loads((Path(job['project_dir']) / 'project.json').read_text(encoding='utf-8'))
        if (receipt.get('story_id') != job['id'] or
                receipt.get('script_sha256') != project['script_sha256']):
            raise ValueError('Delivery does not match the current original story')
        folders.append(folder)
    return write_review_archive(folders, Path(zip_path))


def main(argv=None):
    p=argparse.ArgumentParser(description="Two Worlds all-6 Russian scene jobs, locally")
    p.add_argument("--queue",type=Path,required=True)
    p.add_argument("--models-root",type=Path,default=Path("private_models"))
    p.add_argument("--out",type=Path,default=Path("local_exports"))
    p.add_argument("--images",action="store_true")
    p.add_argument("--voices",action="store_true")
    p.add_argument("--no-music",action="store_true")
    p.add_argument("--no-render",action="store_true")
    p.add_argument("--status",action="store_true")
    p.add_argument("--package",type=Path)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--speaker",default="Serena")
    args=p.parse_args(argv)
    try:
        if args.status:
            result=batch_status(args.queue)
        elif args.package is not None:
            result=package_review(args.queue,args.out,args.package)
        else:
            result=run_batch(args.queue,args.out,args.models_root,
                             images=args.images,voices=args.voices,
                             music=not args.no_music,render=not args.no_render,
                             device=args.device,speaker=args.speaker)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (ValueError,RuntimeError,OSError) as exc:
        print(json.dumps({"status":"BLOCKED","reason":str(exc)},ensure_ascii=False))
        return 2


if __name__=="__main__":
    raise SystemExit(main())
