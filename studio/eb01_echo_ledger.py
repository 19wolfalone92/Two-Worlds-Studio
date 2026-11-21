"""Two private native25 queues: atomic claims, immutable context, no auto reruns.

This queue does not submit Kaggle jobs. Actual source, quality and capacity
must be rechecked by the authorized private execution process at reservation.
Uncertain execution never expires into a new generation by itself.
"""
from __future__ import annotations
import hashlib,json,math,re,sqlite3
from pathlib import Path
from studio.eb01_echo_production_manifest import plan
from studio.eb01_v8_private_mux import safe_private_path

SHA=re.compile(r'^[a-f0-9]{64}$')
class QueueBlocked(ValueError):pass


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class EchoLedger:
    def __init__(self,path:Path,account:str,manifest=None):
        if account not in ('owner','collaborator'):raise QueueBlocked('Unknown independent account')
        doc=manifest or plan()
        expected=plan(profile=doc['engine_profile'],verified_parallel_processes=doc['verified_parallel_processes_per_account'])
        if digest(doc)!=digest(expected):raise QueueBlocked('Manifest differs from immutable canonical source plan')
        self.account=account;self.manifest=doc;self.manifest_sha=digest(doc)
        self.jobs={j['job_id']:j for j in doc['jobs'] if j['account_alias']==account}
        if len(self.jobs)!=15:raise QueueBlocked('Exactly fifteen unique jobs per independent account required')
        path=safe_private_path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.path=path;self.db=sqlite3.connect(path,timeout=15,isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL');path.chmod(0o600)
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY,scene TEXT NOT NULL,segment_index INTEGER NOT NULL,
            context_sha256 TEXT NOT NULL,conditioning_png_sha256 TEXT,state TEXT NOT NULL DEFAULT 'PENDING',nonce TEXT UNIQUE,
            execution_ref TEXT,mp4_sha256 TEXT UNIQUE,tail_png_sha256 TEXT UNIQUE,tail_rgb_sha256 TEXT,
            gpu_seconds REAL,physical_gpu_index INTEGER,UNIQUE(scene,segment_index));
        CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY,job_id TEXT NOT NULL,
            state TEXT NOT NULL,created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        ''')
        # Existing private checkpoints remain readable. Pending preparations
        # from the first schema have no reservations or spent GPU time.
        if 'estimated_gpu_seconds' not in {r[1] for r in self.db.execute('PRAGMA table_info(jobs)')}:
            self.db.execute('ALTER TABLE jobs ADD COLUMN estimated_gpu_seconds REAL')
        try:
            with self.transaction():
                for key,val in {'account':account,'manifest_sha256':self.manifest_sha,'profile':doc['engine_profile']}.items():
                    old=self.db.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
                    if old and old[0]!=val:raise QueueBlocked('Immutable queue account or manifest changed')
                    self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',(key,val))
                for j in self.jobs.values():
                    self.db.execute('INSERT OR IGNORE INTO jobs(job_id,scene,segment_index,context_sha256) VALUES (?,?,?,?)',
                        (j['job_id'],j['scene_id'],j['segment_index'],j['job_context_sha256']))
        except BaseException:self.db.close();raise

    def transaction(self):
        db=self.db
        class Tx:
            def __enter__(self):db.execute('BEGIN IMMEDIATE')
            def __exit__(self,kind,value,tb):db.execute('ROLLBACK' if kind else 'COMMIT')
        return Tx()

    def close(self):self.db.close()

    def reserve(self,job_id,nonce,*,quality_confirmed=False,source_reverified=False,
                fresh_quota_seconds=None,estimated_gpu_seconds=None,conditioning_png_sha256=None,physical_gpu_index=0):
        if quality_confirmed is not True or source_reverified is not True:raise QueueBlocked('Actual quality and private source rechecks required')
        if any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in (fresh_quota_seconds,estimated_gpu_seconds)):
            raise QueueBlocked('Fresh real quota and measured compute estimate required')
        if estimated_gpu_seconds*1.5>fresh_quota_seconds:raise QueueBlocked('Account quota reserve insufficient')
        if type(physical_gpu_index) is not int or physical_gpu_index not in (0,1):raise QueueBlocked('Invalid separate physical card')
        if self.manifest['verified_parallel_processes_per_account']==1 and physical_gpu_index!=0:raise QueueBlocked('Unverified second video card capacity')
        if not isinstance(nonce,str) or not re.fullmatch(r'[a-z0-9-]{8,80}',nonce):raise QueueBlocked('Invalid immutable execution nonce')
        if not SHA.fullmatch(str(conditioning_png_sha256)):raise QueueBlocked('Exact private native reference PNG hash required')
        with self.transaction():
            if self.db.execute("SELECT 1 FROM jobs WHERE state='UNCERTAIN' LIMIT 1").fetchone():
                raise QueueBlocked('Uncertain prior execution must be reconciled, never auto repeated')
            active=self.db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('RESERVED','SUBMITTED')").fetchone()[0]
            if active>=self.manifest['verified_parallel_processes_per_account']:raise QueueBlocked('Verified independent card capacity exhausted')
            if self.db.execute("SELECT 1 FROM jobs WHERE state IN ('RESERVED','SUBMITTED') AND physical_gpu_index=? LIMIT 1",(physical_gpu_index,)).fetchone():
                raise QueueBlocked('This physical card already has an unresolved job')
            active_estimates=self.db.execute("SELECT estimated_gpu_seconds FROM jobs WHERE state IN ('RESERVED','SUBMITTED')").fetchall()
            if any(type(r[0]) not in (int,float) or not math.isfinite(r[0]) or r[0]<=0 for r in active_estimates):
                raise QueueBlocked('Older active reservation requires actual quota reconciliation')
            if (sum(r[0] for r in active_estimates)+estimated_gpu_seconds)*1.5>fresh_quota_seconds:
                raise QueueBlocked('Combined independent-card quota reserve insufficient')
            j=self.jobs.get(job_id)
            row=self.db.execute('SELECT state FROM jobs WHERE job_id=?',(job_id,)).fetchone()
            if j is None or row!=('PENDING',):raise QueueBlocked('Unknown or previously reserved job')
            predecessor=j['predecessor_job_id']
            if predecessor:
                before=self.db.execute('SELECT state,tail_png_sha256 FROM jobs WHERE job_id=?',(predecessor,)).fetchone()
                if before!=('VERIFIED',conditioning_png_sha256):raise QueueBlocked('Exact verified predecessor tail PNG required')
            self.db.execute("UPDATE jobs SET state='RESERVED',nonce=?,conditioning_png_sha256=?,physical_gpu_index=?,estimated_gpu_seconds=? WHERE job_id=?",
                (nonce,conditioning_png_sha256,physical_gpu_index,estimated_gpu_seconds,job_id))
            self.db.execute("INSERT INTO events(job_id,state) VALUES (?,'RESERVED')",(job_id,))

    def mark_submission(self,job_id,nonce,execution_ref):
        if execution_ref is not None and (not isinstance(execution_ref,str) or not re.fullmatch(r'[a-zA-Z0-9_:/.-]{6,160}',execution_ref)):
            raise QueueBlocked('Invalid bounded private execution reference')
        state='SUBMITTED' if execution_ref else 'UNCERTAIN'
        with self.transaction():
            row=self.db.execute('SELECT state,nonce FROM jobs WHERE job_id=?',(job_id,)).fetchone()
            if row!=('RESERVED',nonce):raise QueueBlocked('No matching pending reservation')
            self.db.execute('UPDATE jobs SET state=?,execution_ref=? WHERE job_id=?',(state,execution_ref,job_id))
            self.db.execute('INSERT INTO events(job_id,state) VALUES (?,?)',(job_id,state))

    def record_verified(self,job_id,nonce,clip:Path,tail_png:Path,receipt:dict,*,gpu_seconds,physical_gpu_index):
        from studio.comfy_neural import _probe
        from studio.eb01_motion_artifact_gate import inspect_clip
        from tools.eb01_native_pixel_evidence import decoded_pixel_evidence
        from PIL import Image
        import subprocess
        j=self.jobs.get(job_id)
        if j is None:raise QueueBlocked('Unknown native segment')
        if (type(gpu_seconds) not in (int,float) or not math.isfinite(gpu_seconds) or gpu_seconds<=0 or
            type(physical_gpu_index) is not int or physical_gpu_index not in (0,1)):
            raise QueueBlocked('Real compute and separate physical card measurement required')
        clip=safe_private_path(clip);tail_png=safe_private_path(tail_png)
        mp4=hashlib.sha256(clip.read_bytes()).hexdigest();png=hashlib.sha256(tail_png.read_bytes()).hexdigest()
        meta=_probe(clip);n=j['native_frames']
        if (meta['width']!=576 or meta['height']!=1024 or meta['fps']!='25' or meta['frames']!=n or
            inspect_clip(clip,expected_fps=25)['status']!='NO_NEON_ARTIFACT_FOUND'):
            raise QueueBlocked('Actual original native segment geometry or RGB check failed')
        pixels=decoded_pixel_evidence(clip,frames=n)
        if pixels['unique_decoded_rgb_frames']!=n:raise QueueBlocked('Duplicate or fabricated native frames')
        if (receipt.get('job_context_sha256')!=j['job_context_sha256'] or receipt.get('mp4_sha256')!=mp4 or
            receipt.get('canonical_portrait_sha256')!=j['canonical_portrait_sha256'] or
            receipt.get('canonical_voice_sha256')!=j['canonical_voice_sha256'] or
            receipt.get('audio_conditioned') is not True or receipt.get('source_video_upscaled') is not False):
            raise QueueBlocked('Actual bytes do not match immutable original input provenance')
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(clip),'-vf',f'select=eq(n\\,{n-1})',
            '-vsync','0','-pix_fmt','rgb24','-f','rawvideo','-'],timeout=60)
        with Image.open(tail_png) as im:
            if im.size!=(576,1024) or im.convert('RGB').tobytes()!=raw:raise QueueBlocked('Tail PNG is not exact last native decoded frame')
        rgb=hashlib.sha256(raw).hexdigest()
        with self.transaction():
            row=self.db.execute('SELECT state,nonce,conditioning_png_sha256,physical_gpu_index FROM jobs WHERE job_id=?',(job_id,)).fetchone()
            if (not row or row[0] not in ('SUBMITTED','UNCERTAIN') or row[1]!=nonce or
                row[2]!=receipt.get('conditioning_png_sha256') or row[3]!=physical_gpu_index):
                raise QueueBlocked('No matching actual execution, reference PNG or physical GPU')
            self.db.execute("UPDATE jobs SET state='VERIFIED',mp4_sha256=?,tail_png_sha256=?,tail_rgb_sha256=?,gpu_seconds=?,physical_gpu_index=? WHERE job_id=?",
                (mp4,png,rgb,gpu_seconds,physical_gpu_index,job_id))
            self.db.execute("INSERT INTO events(job_id,state) VALUES (?,'VERIFIED')",(job_id,))

    def snapshot(self,destination:Path):
        destination=safe_private_path(destination)
        if destination.exists():raise FileExistsError('Never overwrite a durable queue checkpoint')
        destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        with sqlite3.connect(destination) as target:self.db.backup(target)
        destination.chmod(0o600)
        return hashlib.sha256(destination.read_bytes()).hexdigest()

    def summary(self):
        return {'account_alias':self.account,'manifest_sha256':self.manifest_sha,
            'counts':dict(self.db.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state')),
            'automatic_dispatch':False,'quality_approved':False,'whole_movie_ready':False}
