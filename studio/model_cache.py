"""Stream attached model shards to fast transient disk before mmap-based loading.

Observed Kaggle mounted-shard paging took 75 minutes while two actual FLUX
frames took under a minute. One bounded sequential copy avoids random remote
page faults, preserves verified hashes and does not duplicate persisted weights.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import tempfile
import time
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
