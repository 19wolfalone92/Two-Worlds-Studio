"""Stream attached model shards to fast transient disk before mmap-based loading.

Observed Kaggle mounted-shard paging took 75 minutes while two actual FLUX
frames took under a minute. One bounded sequential copy avoids random remote
page faults, preserves verified hashes and does not duplicate persisted weights.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path


def stage_cached_tree(source: Path,*,max_bytes=18*1024**3) -> tuple[Path,dict]:
    source=Path(source).resolve()
    if not source.is_dir():raise FileNotFoundError(source)
    files=[p for p in source.rglob('*') if p.is_file() and '.cache' not in p.relative_to(source).parts
           and not p.name.endswith(('.incomplete','.partial'))]
    if any(source not in p.resolve().parents for p in files):
        raise ValueError('A model cache link escapes its declared source')
    size=sum(p.stat().st_size for p in files)
    if not files or size>max_bytes:raise ValueError('Missing model or explicit staging budget exceeded')
    dest=Path(tempfile.mkdtemp(prefix='tw-local-model-'))
    if shutil.disk_usage(dest).free<size+512*1024**2:
        shutil.rmtree(dest)
        raise RuntimeError('Insufficient transient disk for the bounded model staging pass')
    started=time.monotonic()
    def copy(path):
        relative=path.relative_to(source)
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256()
        with path.open('rb') as inp,target.open('xb') as out:
            while data:=inp.read(8*1024**2):
                out.write(data);h.update(data)
        if target.stat().st_size!=path.stat().st_size:raise RuntimeError('Incomplete model copy')
        return {'file':relative.as_posix(),'bytes':target.stat().st_size,'sha256':h.hexdigest()}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:inventory=list(pool.map(copy,files))
        receipt={'bytes':size,'files':inventory,'elapsed_s':round(time.monotonic()-started,3),
                 'strategy':'bounded_stream_copy_before_mmap','persistent_weight_duplicate':False}
        (dest/'local_stage_receipt.json').write_text(json.dumps(receipt,indent=2))
        return dest,receipt
    except BaseException:
        shutil.rmtree(dest,ignore_errors=True)
        raise


def cached_segment(cache: Path, fingerprint: str, destination: Path, render) -> tuple[dict, bool]:
    """Reuse only verified outputs. Lease through the existing private production queue."""
    from studio import production_queue as queue
    from studio.release_gate import sha256

    cache = queue.private_path(cache)
    worker = "render." + uuid.uuid4().hex
    job = "segment." + fingerprint
    with closing(queue.open_queue(cache / "jobs.sqlite3")) as db:
        queue.submit(db, job, "scene_animation", fingerprint)
        artifact = cache / (fingerprint + ".mp4")
        receipt = artifact.with_suffix(".json")
        result = queue.completed_result(db, job)
        if result is not None:
            try:
                data = json.loads(receipt.read_text(encoding="utf-8"))
                verified = (sha256(receipt) == result and data["video_sha256"] == sha256(artifact)
                            and isinstance(data["stats"], dict))
            except (OSError, ValueError, KeyError, TypeError):
                verified = False
            if verified:
                shutil.copyfile(artifact, destination)
                return data["stats"], True
            queue.invalidate_result(db, job, result)
        lease = queue.claim(db, worker, job_id=job, lease_seconds=3600)
        if lease is None:
            raise RuntimeError("Shot cache job is in progress or waiting for a bounded retry")
        try:
            with tempfile.TemporaryDirectory(prefix=".building-", dir=cache) as folder:
                temporary = Path(folder) / "shot.mp4"
                stats = render(temporary)
                data = {"video_sha256": sha256(temporary), "stats": stats}
                proof = temporary.with_suffix(".json")
                proof.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
                shutil.copyfile(temporary, destination)
                temporary.replace(artifact)
                proof.replace(receipt)
                queue.succeed(db, job, worker, sha256(receipt))
        except BaseException as exc:
            reason = "unsafe_input" if isinstance(exc, ValueError) else "unknown_error"
            queue.fail(db, job, worker, reason)
            raise
    return stats, False
