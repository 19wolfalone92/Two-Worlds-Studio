"""Opt-in source-to-diff adviser: produces a checked patch, never applies it.

Only explicitly selected files leave the machine. Git apply --check is read-only
and does not execute generated code. Nothing is committed or published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ALLOWED_SUFFIXES = {'.py', '.js', '.mjs', '.css', '.html', '.json'}
FORBIDDEN_PARTS = {'.git', '.github', '.env', '.venv', 'node_modules',
                   'secrets', 'dist', 'build', 'private', 'credentials'}
MAX_SOURCE_BYTES = 20000
MAX_PATCH_BYTES = 40000


def select_files(source: Path, names: list[str]) -> dict[str, str]:
    source = source.resolve(strict=True)
    if len(names) not in range(1, 4) or len(set(names)) != len(names):
        raise ValueError('Choose 1..3 distinct existing source files')
    found = {}
    for name in names:
        p = PurePosixPath(name)
        if (p.is_absolute() or '.' in p.parts or '..' in p.parts or
            any(part.lower() in FORBIDDEN_PARTS or part.startswith('.') for part in p.parts)
            or p.suffix.lower() not in ALLOWED_SUFFIXES):
            raise ValueError('Unsafe requested file path')
        target = source.joinpath(*p.parts)
        if not target.is_file() or target.is_symlink() or source not in target.resolve().parents:
            raise ValueError('File must be a regular file inside the approved repository')
        data = target.read_bytes()
        if len(data) > MAX_SOURCE_BYTES:
            raise ValueError('Source file too large to disclose')
        found[p.as_posix()] = data.decode('utf-8')
    return found


def validate_patch(patch: str, selected: set[str]) -> None:
    if not patch.startswith('diff --git a/') or len(patch.encode('utf-8')) > MAX_PATCH_BYTES:
        raise ValueError('Expected one small unified git patch')
    if '\x00' in patch or re.search(r'(?im)^(?:rename |copy |new file mode|deleted file mode|old mode|new mode|GIT binary patch)', patch):
        raise ValueError('No binary/mode/create/rename/delete operations permitted')
    headers = re.findall(r'^diff --git a/(\S+) b/(\S+)$', patch, flags=re.M)
    if not headers or any(a != b or a not in selected for a, b in headers):
        raise ValueError('Patch attempts to modify files outside the approved selection')
    for token in re.findall(r'^(?:---|\+\+\+) (\S+)', patch, flags=re.M):
        if not token.startswith(('a/', 'b/')) or token[2:] not in selected:
            raise ValueError('Patch contains unexpected file paths')


def propose(source: Path, names: list[str], issue: str, *, base_url: str,
            model: str, output: Path, api_key: str = '', send_source: bool = False) -> dict:
    if not send_source:
        raise ValueError('Must explicitly opt in with --send-source')
    if not 5 <= len(issue) <= 800:
        raise ValueError('Describe the bounded issue in 5..800 characters')
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('Patch output must be outside the source checkout')
    files = select_files(source, names)
    parts = urlsplit(base_url)
    if not parts.netloc or parts.scheme not in ('https', 'http'):
        raise ValueError('Invalid OpenAI-compatible endpoint')
    if parts.scheme == 'http' and parts.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Remote HTTP is prohibited')
    if any((parts.username, parts.password, parts.query, parts.fragment)):
        raise ValueError('Do not include credentials or query parameters in URL')
    messages = [
        {'role': 'system', 'content': 'Output only a valid git unified diff, starting with diff --git. Touch only the supplied paths; no new files, no deletion, no mode changes. Do not claim a test was run. Minimal fix only.'},
        {'role': 'user', 'content': 'Problem: ' + issue + '\n\nFILES:\n' +
         '\n'.join(f'===== {name} =====\n{value}' for name, value in files.items())},
    ]
    payload = {'model': model, 'max_tokens': 3200, 'temperature': 0,
               'messages': messages}
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    req = Request(base_url.rstrip('/') + '/chat/completions',
                  data=json.dumps(payload).encode('utf-8'), method='POST', headers=headers)
    with urlopen(req, timeout=75) as response:
        patch = str(json.load(response)['choices'][0]['message']['content']).strip() + '\n'
    validate_patch(patch, set(files))
    with __import__('tempfile').NamedTemporaryFile('w', encoding='utf-8', suffix='.patch') as t:
        t.write(patch)
        t.flush()
        check = subprocess.run(['git', 'apply', '--check', '--whitespace=error', t.name],
                               cwd=source, text=True, capture_output=True, timeout=20, check=False)
    if check.returncode != 0:
        raise ValueError('Patch does not apply cleanly: ' + check.stderr[:300])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(patch, encoding='utf-8')
    return {'status': 'proposal_only', 'files': list(files),
            'model': model, 'patch_sha256': hashlib.sha256(patch.encode('utf-8')).hexdigest(),
            'git_apply_check': 'passed', 'applied': False, 'tests_run': False,
            'source_sent': True, 'cost_not_guaranteed_zero': True}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--file', action='append', required=True, dest='files')
    p.add_argument('--issue', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--send-source', action='store_true')
    p.add_argument('--base-url', default=os.getenv('WOLF_LLM_BASE_URL', 'http://127.0.0.1:11434/v1'))
    p.add_argument('--model', default=os.getenv('WOLF_LLM_MODEL', 'qwen2.5-coder:7b'))
    args = p.parse_args()
    result = propose(args.source, args.files, args.issue,
                     base_url=args.base_url, model=args.model, output=args.output,
                     api_key=os.getenv('WOLF_LLM_API_KEY', ''), send_source=args.send_source)
    report = args.output.with_suffix('.report.json')
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
