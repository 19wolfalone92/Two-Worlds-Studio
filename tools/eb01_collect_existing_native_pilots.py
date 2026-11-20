"""Read only the two original November 16 native jobs; never submit a GPU kernel."""
from __future__ import annotations
import argparse
import json
import os
import tempfile
from pathlib import Path
from tools.eb01_native_two_account_pilot import run_pilot,call,prepare,PilotBlocked
from tools.eb01_kaggle2_auth_probe import query

SOURCE_RUN = "43918550500"
SOURCE_REVISION = "75115ac6e6b63dae25e879ea86d32cf015bda49c"


def main():
    p=argparse.ArgumentParser();p.add_argument('--account',choices=('owner','collaborator'),required=True)
    args=p.parse_args();os.umask(0o077)
    token=os.environ.get('EB01_ACCOUNT_TOKEN','')
    with tempfile.TemporaryDirectory(dir=os.environ['RUNNER_TEMP'],prefix='eb01-native-private-') as t:
        root=Path(t)
        report=run_pilot(args.account,token,root,os.environ['GITHUB_RUN_ID'],SOURCE_REVISION,
                         2,collect_run=SOURCE_RUN,poll_seconds=60)
        # Read the actual submitted notebook flags; no private code or names
        # enter the numerical report. A successful push is not GPU allocation.
        state,handle=query(token)
        if state=='AUTHENTICATED_READ_ONLY' and handle:
            import hashlib
            nonce=hashlib.sha256(('eb01-native-v1-'+SOURCE_RUN+'-'+args.account).encode()).hexdigest()[:12]
            kernel=handle+'/eb01-native-pilot-'+nonce
            dest=root/'metadata';dest.mkdir(mode=0o700)
            try:
                call(token,['kaggle','kernels','pull',kernel,'-p',str(dest),'-m'],timeout=90)
                meta=json.loads((dest/'kernel-metadata.json').read_text())
                report['submitted_metadata']={k:meta.get(k) for k in ('is_private','enable_gpu','enable_internet','machine_shape')}
            except (PilotBlocked,OSError,ValueError) as exc:
                report['metadata_status']='BLOCKED_'+(str(exc) if isinstance(exc,PilotBlocked) else type(exc).__name__)
            # If the service still says queued after days, test output access
            # exactly once. No polling storm and no replacement render.
            if report.get('kaggle_state') in ('QUEUED','RUNNING'):
                out=root/'readback-check';out.mkdir(mode=0o700)
                try:
                    call(token,['kaggle','kernels','output',kernel,'-p',str(out),'--force'],timeout=120)
                    report['queued_output_mp4_present']=(out/'S02_PILOT33.mp4').is_file()
                    report['queued_output_files_count']=len([x for x in out.rglob('*') if x.is_file()])
                except PilotBlocked as exc:
                    report['queued_output_probe']='BLOCKED_'+str(exc)
    report['collection_only_no_new_gpu']=True
    path=Path(os.environ['RUNNER_TEMP'])/('EB01_NATIVE_'+args.account+'_SAFE.json')
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('EB01_EXISTING_NATIVE_PILOT_COLLECTION',json.dumps(report,sort_keys=True))
    # Completed read-only classification may legitimately report no footage.
    return 0

if __name__=='__main__':raise SystemExit(main())
