"""Durable local CPU-first job queue for future multi-provider studio dispatch.

It NEVER calls external services or stores raw prompts, tokens, private art,
script bytes, identities or raw provider error bodies. It is safe to resume on
a single machine with a persistent private SQLite volume; GitHub Actions
temporary runners do NOT offer cross-run persistence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,110}$")
SHA = re.compile(r"^[0-9a-f]{64}$")
KINDS = {"image_generation", "image_qc", "depth_map", "segmentation",
         "actor_animation", "scene_animation", "russian_tts", "music",
         "subtitle_alignment", "montage", "release_qa", "publish_staging"}
REASONS = {"http_429", "quota_exhausted", "network_error", "provider_unavailable",
           "dependency_missing", "invalid_response", "unsafe_input",
           "unknown_error"}
MAX_ATTEMPTS = 4


def private_path(path: Path) -> Path:
    path = path.resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Private job state cannot be stored in the public source repository")
    return path


def open_queue(path: Path) -> sqlite3.Connection:
    path = private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=15, isolation_level=None)
    db.execute("PRAGMA busy_timeout=15000")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""
    CREATE TABLE IF NOT EXISTS studio_jobs (
      job_id TEXT PRIMARY KEY,
      kind TEXT NOT NULL,
      payload_sha256 TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'waiting',
      attempts INTEGER NOT NULL DEFAULT 0,
      retry_at REAL NOT NULL DEFAULT 0,
      lease_owner TEXT,
      lease_until REAL,
      result_sha256 TEXT,
      failure_reason TEXT,
      updated_at REAL NOT NULL
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS studio_jobs_ready ON studio_jobs(status,retry_at)")
    return db


def submit(db: sqlite3.Connection, job_id: str, kind: str,
           payload_sha256: str, *, now: float | None = None) -> dict:
    if not isinstance(job_id,str) or not IDENTIFIER.fullmatch(job_id):
        raise ValueError("Unsafe job ID")
    if kind not in KINDS or not isinstance(payload_sha256,str) or not SHA.fullmatch(payload_sha256):
        raise ValueError("Unknown task kind or invalid opaque payload hash")
    stamp = time.time() if now is None else now
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute("SELECT kind,payload_sha256 FROM studio_jobs WHERE job_id=?",
                         (job_id,)).fetchone()
        if row:
            if row != (kind, payload_sha256):
                raise ValueError("Existing job ID refers to different input; cannot overwrite")
            inserted = False
        else:
            db.execute("INSERT INTO studio_jobs(job_id,kind,payload_sha256,updated_at) VALUES(?,?,?,?)",
                       (job_id,kind,payload_sha256,stamp))
            inserted = True
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    return {"job_id":job_id,"enqueued":inserted,"status":"waiting" if inserted else "existing"}


def claim(db: sqlite3.Connection, worker_id: str, *,
          now: float | None = None, lease_seconds: int = 600) -> dict | None:
    if not isinstance(worker_id,str) or not IDENTIFIER.fullmatch(worker_id):
        raise ValueError("Unsafe worker ID")
    if type(lease_seconds) is not int or not 30 <= lease_seconds <= 3600:
        raise ValueError("Lease duration must be 30..3600 seconds")
    stamp=time.time() if now is None else now
    db.execute("BEGIN IMMEDIATE")
    try:
        row=db.execute("""
          SELECT job_id,kind,payload_sha256,attempts
          FROM studio_jobs
          WHERE (status='waiting' AND retry_at<=?)
            OR (status='running' AND lease_until<=?)
          ORDER BY retry_at ASC, updated_at ASC, job_id ASC LIMIT 1
        """,(stamp,stamp)).fetchone()
        if not row:
            db.execute("COMMIT")
            return None
        job_id,kind,payload_sha256,attempts=row
        if attempts >= MAX_ATTEMPTS:
            db.execute("UPDATE studio_jobs SET status='failed', failure_reason='unknown_error', updated_at=? WHERE job_id=?",
                       (stamp,job_id))
            db.execute("COMMIT")
            return None
        db.execute("""
          UPDATE studio_jobs SET status='running', attempts=attempts+1,
            lease_owner=?, lease_until=?, updated_at=? WHERE job_id=?
        """,(worker_id,stamp+lease_seconds,stamp,job_id))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    return {"job_id":job_id,"kind":kind,"payload_sha256":payload_sha256,
            "attempt":attempts+1,"lease_owner":worker_id,"lease_until":stamp+lease_seconds}


def _require_owned(db: sqlite3.Connection, job_id: str, worker_id: str,
                   stamp: float) -> int:
    row=db.execute("SELECT status,lease_owner,lease_until,attempts FROM studio_jobs WHERE job_id=?",
                   (job_id,)).fetchone()
    if not row or row[0]!="running" or row[1]!=worker_id or row[2]<=stamp:
        raise ValueError("Job is not held by this active worker lease")
    return int(row[3])


def succeed(db: sqlite3.Connection, job_id: str, worker_id: str,
            result_sha256: str, *, now: float | None = None) -> dict:
    if not isinstance(result_sha256,str) or not SHA.fullmatch(result_sha256):
        raise ValueError("Result requires a verified SHA256 hash")
    stamp=time.time() if now is None else now
    db.execute("BEGIN IMMEDIATE")
    try:
        _require_owned(db,job_id,worker_id,stamp)
        db.execute("""UPDATE studio_jobs SET status='complete', result_sha256=?,
          lease_owner=NULL, lease_until=NULL, updated_at=? WHERE job_id=?""",
          (result_sha256,stamp,job_id))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    return {"job_id":job_id,"status":"complete","result_sha256":result_sha256}


def fail(db: sqlite3.Connection, job_id: str, worker_id: str, reason: str,
         *, now: float | None = None, retry_after: int | None = None) -> dict:
    if reason not in REASONS:
        raise ValueError("Failure reason must be a safe enum, not raw provider text")
    if retry_after is not None and (type(retry_after) is not int or not 0 <= retry_after <= 86400):
        raise ValueError("Invalid Retry-After")
    stamp=time.time() if now is None else now
    db.execute("BEGIN IMMEDIATE")
    try:
        attempts=_require_owned(db,job_id,worker_id,stamp)
        terminal=reason in {"unsafe_input","invalid_response"} or attempts >= MAX_ATTEMPTS
        state="failed" if terminal else "waiting"
        delay=max(min(3600,60*2**(attempts-1)),retry_after or 0)
        if reason=="quota_exhausted":
            delay=max(delay,86400)  # Free daily allowance; do not churn requests
        db.execute("""UPDATE studio_jobs SET status=?, retry_at=?,
          lease_owner=NULL, lease_until=NULL, failure_reason=?,updated_at=?
          WHERE job_id=?""",(state,stamp+delay,reason,stamp,job_id))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    return {"job_id":job_id,"status":state,"retry_after_seconds":0 if terminal else delay,
            "reason":reason,"attempts":attempts}


def status(db: sqlite3.Connection) -> dict:
    rows=db.execute("""SELECT kind,status,COUNT(*) FROM studio_jobs
                       GROUP BY kind,status ORDER BY kind,status""").fetchall()
    return {"jobs":[{"kind":kind,"status":state,"count":count} for kind,state,count in rows],
            "no_raw_prompts_or_tokens":True,"provider_execution_implemented":False}


def enqueue_routes(db: sqlite3.Connection, plan: dict, *, now: float | None = None) -> dict:
    """Persist eligible free-only dry-run scenes. Does NOT execute GPU/API calls."""
    if (not isinstance(plan,dict) or plan.get("schema")!=1
            or plan.get("safe_free_only") is not True
            or plan.get("auto_publish") is not False):
        raise ValueError("Only safe free-only non-publishing route plans can be queued")
    episode=plan.get("episode")
    if not isinstance(episode,str) or not IDENTIFIER.fullmatch(episode):
        raise ValueError("Safe episode ID required")
    source=plan.get("source_prompt_pack_sha256")
    if not isinstance(source,str) or not SHA.fullmatch(source):
        raise ValueError("Source prompt pack SHA256 required")
    rows=plan.get("routes")
    if not isinstance(rows,list) or len(rows)>500:
        raise ValueError("Invalid planned scenes")
    count=0
    skipped=0
    for row in rows:
        if not isinstance(row,dict) or row.get("automatic_submission") is not False:
            raise ValueError("Cannot queue auto-submitting provider jobs")
        provider=row.get("provider")
        if provider is None:
            skipped+=1
            continue
        if provider not in {"cloudflare_image","horde"}:
            raise ValueError("Unreviewed provider is not allowed")
        if provider=="horde" and (row.get("private") is not False
                                  or row.get("reference_count")!=0):
            raise ValueError("Private or referenced art must never use volunteer Horde")
        scene=row.get("scene")
        if not isinstance(scene,str) or not IDENTIFIER.fullmatch(scene):
            raise ValueError("Invalid planned scene")
        fingerprint=row.get("prompt_sha256")
        if not isinstance(fingerprint,str) or not SHA.fullmatch(fingerprint):
            raise ValueError("Missing deterministic prompt hash")
        job_id=f"{episode}.{scene}.art"
        digest=hashlib.sha256((source+":"+provider+":"+fingerprint).encode()).hexdigest()
        submitted=submit(db,job_id,"image_generation",digest,now=now)
        count+=int(submitted["enqueued"])
    return {"new_jobs":count,"skipped_unavailable":skipped,"external_calls":0,
            "scripts_modified":False,"published":False}


def main(argv=None):
    p=argparse.ArgumentParser(description="Private durable studio task ledger; no external provider calls")
    p.add_argument("--database",type=Path,default=Path("/tmp/two_worlds_studio/jobs.sqlite3"))
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--status",action="store_true")
    modes.add_argument("--submit",type=str)
    modes.add_argument("--route-plan",type=Path)
    p.add_argument("--kind",choices=sorted(KINDS))
    p.add_argument("--payload-sha256")
    args=p.parse_args(argv)
    with open_queue(args.database) as db:
        if args.status:
            result=status(db)
        elif args.route_plan:
            plan=json.loads(args.route_plan.read_text(encoding="utf-8"))
            result=enqueue_routes(db,plan)
        else:
            if not args.kind or not args.payload_sha256:
                p.error("--submit requires --kind and --payload-sha256")
            result=submit(db,args.submit,args.kind,args.payload_sha256)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
