"""Read-only Kaggle capability check; return private diagnostics as sealed bytes."""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
from studio.sealed_artifacts import seal


def main():
    out = Path(os.environ['RUNNER_TEMP']) / 'tw-batch03-preflight'
    out.mkdir(mode=0o700)
    results = {}
    for name, args in (
        ('cli_version', ['kaggle', '--version']),
        ('quota', ['kaggle', 'quota']),
        ('datasets', ['kaggle', 'datasets', 'list', '--mine', '--csv']),
        ('kernels', ['kaggle', 'kernels', 'list', '--mine', '--csv']),
        ('public_flux_models', ['kaggle', 'models', 'list', '--search', 'FLUX.2-klein-4B', '--csv']),
        ('public_qwen_models', ['kaggle', 'models', 'list', '--search', 'Qwen3-TTS', '--csv']),
    ):
        try:
            run = subprocess.run(args, capture_output=True, text=True, timeout=90)
            # Never write an unmasked credential, even if a CLI error echoes it.
            secret = os.environ.get('KAGGLE_API_TOKEN', '')
            clean = lambda s: s.replace(secret, '[REDACTED]') if secret else s
            results[name] = {'returncode': run.returncode,
                             'stdout': clean(run.stdout), 'stderr': clean(run.stderr)}
            print(name + ': ' + ('PASS' if run.returncode == 0 else 'FAILED'), flush=True)
        except subprocess.TimeoutExpired:
            results[name] = {'returncode': -1, 'error': 'TIMEOUT'}
            print(name + ': TIMEOUT', flush=True)
    results['gpu_allocated'] = False
    results['used_paid_services'] = False
    # Discover the real URL suffix returned by Kaggle, never resubmit a possibly
    # running GPU job merely because its requested slug differs from the title.
    import csv, io
    import re
    kernels = results.get('kernels', {})
    if kernels.get('returncode') == 0:
        candidates = [r['ref'] for r in csv.DictReader(io.StringIO(kernels['stdout']))
                      if 'batch03' in r.get('ref', '') or 'russian-story' in r.get('ref', '')]
        requested=json.loads(Path('.github/batch03_requests/request.json').read_text()).get('kernels',[])
        if not isinstance(requested,list) or len(requested)>5 or any(
                not isinstance(h,str) or not re.fullmatch(r'wolf19loner92/tw-batch03-[a-z0-9-]+',h)
                for h in requested):
            raise ValueError('Invalid explicit read-only kernel diagnostic request')
        for handle in list(dict.fromkeys(requested+candidates[:5])):
            proc = subprocess.run(['kaggle', 'kernels', 'status', handle],
                                  capture_output=True, text=True, timeout=90)
            results['status:' + handle] = {'returncode': proc.returncode,
                                           'stdout': proc.stdout, 'stderr': proc.stderr}
    private = out / 'diagnostics.json'
    private.write_text(json.dumps(results, indent=2))
    context = {'purpose': 'batch03-account-preflight', 'request': '20261104-v1'}
    seal(private, Path('.github/batch03_requests/recipient_public.pem'),
         Path('dist/private_batch03/account_preflight.twenc'), context)
    private.unlink()


if __name__ == '__main__':
    main()
