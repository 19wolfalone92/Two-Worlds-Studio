"""Private durable scene queues; never launches a GPU or approves a movie.

Each account owns one independent SQLite file bound to the SAME immutable
15/15 manifest. Submission uncertainty is persisted and never expires into
an automatic retry. Actual media acceptance remains a separate requirement.
"""
from __future__ import annotations
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from studio.eb01_dual_kaggle_scheduler import ACCOUNT_IDS, Worker, worker_gates, independent_slots, _least_load_assignment, QUOTA_FACTOR
from studio.eb01_v8_40s_native_motion_plan import plan
from studio.eb01_v8_private_mux import safe_private_path

SHA=re.compile(r'^[a-f0-9]{64}$')

class LedgerBlocked(ValueError):pass


def production_manifest():
    scenes=plan()['scenes'];owners=_least_load_assignment(scenes,list(ACCOUNT_IDS))
    jobs=[]
    for s in scenes:
        for i in range(s['native_segment_count']):
            jobs.append({'job_id':f'{s["scene_id"]}_{i:02d}','scene':s['scene_id'],
                'index':i,'account':owners[s['scene_id']],
                'predecessor':f'{s["scene_id"]}_{i-1:02d}' if i else None,
                'canonical_portrait_sha256':s['canonical_reference_sha256'],
                'canonical_voice_sha256':s['canonical_voice_sha256'],
                'canonical_prompt_sha256':s['canonical_prompt_sha256'],
                'native_dimensions':[576,1024],'native_frames':33})
    digest=hashlib.sha256(json.dumps(jobs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'schema':1,'manifest_sha256':digest,'jobs':jobs,
            'takes_per_account':{a:sum(j['account']==a for j in jobs) for a in ACCOUNT_IDS},
            'production_authorized':False,'automatic_gpu_dispatch':False,
            'native_actor_and_russian_lipsync_approval_required':True,
            'remaining_quota_per_account_required':True}


class Ledger:
    def __init__(self,path:Path,account:str):
        if account not in ACCOUNT_IDS:raise LedgerBlocked('Unknown account')
        path=safe_private_path(path);path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        self.account=account
        self.connection=sqlite3.connect(path,timeout=15,isolation_level=None)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        path.chmod(0o600)
        self.connection.executescript('''
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY,scene TEXT NOT NULL,
          segment_index INTEGER NOT NULL,state TEXT NOT NULL DEFAULT 'PENDING',
          nonce TEXT UNIQUE,kernel_ref TEXT,mp4_sha256 TEXT,next_png_sha256 TEXT,
          gpu_seconds REAL,UNIQUE(scene,segment_index));
        CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY,job_id TEXT NOT NULL,
          state TEXT NOT NULL,created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        ''')
        doc=production_manifest()
        try:
            with self.transaction():
                for key,value in {'account':account,'manifest_sha256':doc['manifest_sha256']}.items():
                    old=self.connection.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
                    if old and old[0]!=value:raise LedgerBlocked('Immutable account or manifest mismatch')
                    self.connection.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',(key,value))
                for j in doc['jobs']:
                    if j['account']==account:
                        self.connection.execute('INSERT OR IGNORE INTO jobs(job_id,scene,segment_index) VALUES (?,?,?)',(j['job_id'],j['scene'],j['index']))
        except BaseException:
            self.connection.close();raise

    def transaction(self):
        db=self.connection
        class Transaction:
            def __enter__(self):db.execute('BEGIN IMMEDIATE')
            def __exit__(self,kind,value,tb):db.execute('ROLLBACK' if kind else 'COMMIT')
        return Transaction()

    def close(self):self.connection.close()

    def reserve(self,job_id,nonce,worker:Worker,*,predecessor_png_sha256=None):
        if (worker.account_id!=self.account or worker_gates(worker) or independent_slots(worker)<1
                or worker.actual_gpu_seconds_per_33f*QUOTA_FACTOR>worker.available_quota_minutes*60):
            raise LedgerBlocked('Native quality, account, capacity or quota gate blocked')
        if not isinstance(nonce,str) or not re.fullmatch(r'[a-z0-9-]{8,80}',nonce):
            raise LedgerBlocked('Invalid immutable nonce')
        with self.transaction():
            row=self.connection.execute('SELECT scene,segment_index,state FROM jobs WHERE job_id=?',(job_id,)).fetchone()
            if not row or row[2]!='PENDING':raise LedgerBlocked('Take already reserved or unknown')
            if self.connection.execute("SELECT 1 FROM jobs WHERE state IN ('RESERVED','SUBMITTED','UNCERTAIN') LIMIT 1").fetchone():
                raise LedgerBlocked('Account has an unresolved generation')
            if row[1]:
                prior=self.connection.execute('SELECT state,next_png_sha256 FROM jobs WHERE scene=? AND segment_index=?',(row[0],row[1]-1)).fetchone()
                if not prior or prior[0]!='VERIFIED' or not SHA.fullmatch(str(predecessor_png_sha256)) or prior[1]!=predecessor_png_sha256:
                    raise LedgerBlocked('Exact verified predecessor PNG required')
            elif predecessor_png_sha256 is not None:raise LedgerBlocked('First take requires canonical portrait')
            self.connection.execute("UPDATE jobs SET state='RESERVED',nonce=? WHERE job_id=?",(nonce,job_id))
            self.connection.execute("INSERT INTO events(job_id,state) VALUES (?,'RESERVED')",(job_id,))

    def mark_submission(self,job_id,nonce,kernel_ref):
        if kernel_ref is not None and not re.fullmatch(r'[a-zA-Z0-9_-]{2,30}/[a-z0-9-]{6,80}',kernel_ref):
            raise LedgerBlocked('Invalid private kernel ref')
        state='SUBMITTED' if kernel_ref else 'UNCERTAIN'
        with self.transaction():
            row=self.connection.execute('SELECT state,nonce FROM jobs WHERE job_id=?',(job_id,)).fetchone()
            if row!=('RESERVED',nonce):raise LedgerBlocked('No matching reservation')
            self.connection.execute('UPDATE jobs SET state=?,kernel_ref=? WHERE job_id=?',(state,kernel_ref,job_id))
            self.connection.execute('INSERT INTO events(job_id,state) VALUES (?,?)',(job_id,state))

    def record_verified(self,job_id,nonce,clip:Path,tail_png:Path,mp4_sha256,png_sha256,*,gpu_seconds):
        """Record already accepted media, rechecking actual video/tail bytes.

        No raw JSON receipt can complete a take. Production quality remains
        blocked independently until actor/lipsync approval is actually supplied.
        """
        import math,subprocess
        from PIL import Image
        from studio.eb01_v8_private_mux import native33
        from studio.eb01_motion_artifact_gate import inspect_clip
        if type(gpu_seconds) not in (int,float) or not math.isfinite(gpu_seconds) or gpu_seconds<=0:
            raise LedgerBlocked('Actual compute measurement required')
        for path,expected in ((clip,mp4_sha256),(tail_png,png_sha256)):
            path=safe_private_path(path)
            if not SHA.fullmatch(str(expected)) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise LedgerBlocked('Original video or tail checksum mismatch')
        native33(clip)
        if inspect_clip(clip).get('status')!='NO_NEON_ARTIFACT_FOUND':raise LedgerBlocked('Corrupt original remains quarantined')
        raw=subprocess.run(['ffmpeg','-v','error','-i',str(clip),'-vf','select=eq(n\\,32)',
                            '-vsync','0','-pix_fmt','rgb24','-f','rawvideo','-'],capture_output=True,timeout=60)
        with Image.open(tail_png) as im:
            if raw.returncode or im.size!=(576,1024) or im.convert('RGB').tobytes()!=raw.stdout:
                raise LedgerBlocked('Tail PNG is not exact final native frame')
        with self.transaction():
            row=self.connection.execute('SELECT state,nonce FROM jobs WHERE job_id=?',(job_id,)).fetchone()
            if not row or row[0] not in ('SUBMITTED','UNCERTAIN') or row[1]!=nonce:
                raise LedgerBlocked('No matching submitted take')
            self.connection.execute("UPDATE jobs SET state='VERIFIED',mp4_sha256=?,next_png_sha256=?,gpu_seconds=? WHERE job_id=?",(mp4_sha256,png_sha256,gpu_seconds,job_id))
            self.connection.execute("INSERT INTO events(job_id,state) VALUES (?,'VERIFIED')",(job_id,))

    def summary(self):
        return {'account':self.account,'counts':dict(self.connection.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state')),
                'automatic_dispatch':False,'movie_approved':False}
