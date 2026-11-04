"""Recover a private production return, including authenticated large-file parts."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path
from studio.sealed_artifacts import unseal
from studio.kaggle_shorts_worker import _safe_extract


def recover(artifact: Path,key: Path,nonce: str,destination: Path) -> dict:
    if not re.fullmatch(r'[a-z0-9-]{6,64}',nonce):raise ValueError('Invalid expected production nonce')
    destination.mkdir(parents=True,exist_ok=True)
    archive=destination/'result.zip'
    if archive.exists():raise FileExistsError('This return has already been recovered')
    with tempfile.TemporaryDirectory(prefix='tw-return-') as tmp:
        staging=Path(tmp)
        with zipfile.ZipFile(artifact) as z:
            for info in z.infolist():
                name=info.filename
                if (Path(name).name!=name or not re.fullmatch(
                        r'(production_result(?:\.part[0-9]{4})?\.twenc|return_parts\.json)',name)
                        or info.file_size>160*1024**2):
                    raise ValueError('Unexpected file in the encrypted artifact transport')
            z.extractall(staging)
        manifest=staging/'return_parts.json'
        context={'purpose':'batch03-production','nonce':nonce}
        if manifest.exists():
            doc=json.loads(manifest.read_text())
            count=doc.get('count');fingerprint=doc.get('archive_sha256','')
            if (doc.get('schema')!=1 or doc.get('nonce')!=nonce or type(count) is not int
                    or not 1<=count<=32 or not re.fullmatch(r'[0-9a-f]{64}',fingerprint)
                    or [p.get('index') for p in doc.get('parts',[])]!=list(range(count))):
                raise ValueError('Invalid authenticated part manifest')
            with archive.open('xb') as output:
                for index,part in enumerate(doc['parts']):
                    expected=f'production_result.part{index:04d}.twenc'
                    if part.get('file')!=expected:raise ValueError('Incorrect part order or filename')
                    decoded=staging/f'decoded_{index}.bin'
                    unseal(staging/expected,key,decoded,{**context,'part':index,'count':count,
                                                       'archive_sha256':fingerprint})
                    with decoded.open('rb') as stream:
                        while chunk:=stream.read(8*1024**2):output.write(chunk)
                    decoded.unlink()
            with archive.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=fingerprint:
                archive.unlink();raise ValueError('Reassembled private return checksum mismatch')
        else:
            unseal(staging/'production_result.twenc',key,archive,context)
    with zipfile.ZipFile(archive) as z:
        _safe_extract(z,destination/'files')
    return {'recovered':True,'bytes':archive.stat().st_size,'output':str(destination),
            'files':[str(p.relative_to(destination)) for p in (destination/'files').rglob('*') if p.is_file()]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True);p.add_argument('--private-key',type=Path,required=True)
    p.add_argument('--nonce',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(recover(a.artifact,a.private_key,a.nonce,a.out),indent=2))
