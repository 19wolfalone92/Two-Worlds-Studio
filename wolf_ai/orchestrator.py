"""Finite, reproducible project checks; no unapproved writes or publication.

Use as a CLI in GitHub Actions or locally. Task state and a report survive as
artifacts. There is no service, infinite loop, automatic paid API use or hidden
attempt to reset platform allowances.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

OWNER = '19wolfalone92'
PROJECTS = {
    'studio': 'Two-Worlds-Studio',
    'shaira': 'Shaira',
    'williams': 'Williams',
    'williams-ai': 'Williams-AI',
    'english-child': 'English_Child',
}
SECRET = re.compile(r'(?i)(bearer\s+|api[_-]?key\s*[=:]\s*|token\s*[=:]\s*|password\s*[=:]\s*)([^\s\"\']+)')
MAX_LOG_BYTES = 15000


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    timeout: int = 180


def recipes(project: str, mode: str) -> tuple[Step, ...]:
    if project not in PROJECTS:
        raise ValueError('Unrecognized project; never execute arbitrary repository names')
    if mode not in {'audit', 'test'}:
        raise ValueError('Mode must be audit or test')
    # All recipes are fixed by developers, not supplied by issue bodies or AI.
    syntax_code = (
        "from pathlib import Path; import ast; "
        "files=[p for p in Path('.').rglob('*.py') "
        "if not ({'.git','.venv','venv','node_modules'} & set(p.parts))]; "
        "errors=[]; "
        "[(errors.append(str(p)+': '+str(e)) if (e:=check(p)) else None) "
        "for p in files for check in [lambda f: syntax_error(f)]]; "
        "print('Parsed Python files:',len(files)); "
        "[print(e) for e in errors[:30]]; "
        "raise SystemExit(bool(errors))"
    )
    # Parser only: reads source without importing project code or writing .pyc.
    syntax_code = "def syntax_error(p):\n try: ast.parse(p.read_text(encoding='utf-8'), filename=str(p))\n except (SyntaxError,UnicodeError) as e: return e\n" + syntax_code
    syntax = (Step('python-syntax', (sys.executable, '-B', '-c', syntax_code), 150),)
    if project == 'studio':
        steps = (
            syntax[0],
            Step('free-readiness', (sys.executable, '-m', 'studio.free_studio'), 45),
        )
        if mode == 'test':
            steps += tuple(Step('studio-test-' + test,
                        (sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', test + '.py', '-v'), 180)
                        for test in ('test_production_queue', 'test_studio_v3_free', 'test_subpixel_cinematic'))
        return steps
    if project == 'shaira':
        steps = (Step('javascript-syntax', ('npm', 'run', 'check'), 300),)
        return steps  # `check` already includes npm test; no package install needed.
    if project == 'english-child':
        # Read-only by default; the full Android build is already owned by its CI.
        return (Step('project-tree', (sys.executable, '-c',
                  "from pathlib import Path; p=Path('.'); print('Java:',len(list(p.rglob('*.java'))),'Kotlin:',len(list(p.rglob('*.kt'))),'Gradle:',len(list(p.rglob('*.gradle*'))))"), 45),)
    if project == 'williams':
        return syntax  # Never place orders; never change this bot from another chat.
    if project == 'williams-ai':
        return syntax  # No GPU/model runs without separate explicit approval.
    raise AssertionError(project)


def clean_log(data: str) -> str:
    data = SECRET.sub(lambda m: m.group(1) + '[REDACTED]', data)
    return data[:MAX_LOG_BYTES] + ('\n[TRUNCATED]' if len(data) > MAX_LOG_BYTES else '')


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def run(project: str, mode: str, source: Path, output: Path, *, execute: bool = False,
        budget_seconds: int = 600) -> dict:
    steps = recipes(project, mode)
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise ValueError('Source directory does not exist')
    if output == source or source in output.parents:
        raise ValueError('Reports must be outside target source tree')
    if not 15 <= budget_seconds <= 1800:
        raise ValueError('Budget must be 15..1800 seconds')
    issued = datetime.now(timezone.utc).isoformat()
    result = {'schema': 1, 'product': 'wolf-ai', 'project': project,
              'repository': OWNER + '/' + PROJECTS[project], 'mode': mode,
              'started_utc': issued, 'execute': execute,
              'no_remote_mutations': True, 'no_publication': True, 'no_paid_api': True,
              'budget_seconds': budget_seconds, 'steps': []}
    start = time.monotonic()
    for step in steps:
        entry = {'name': step.name, 'command': list(step.argv), 'status': 'planned',
                 'timeout_seconds': step.timeout}
        if execute:
            remaining = budget_seconds - (time.monotonic() - start)
            if remaining < 1:
                entry['status'] = 'skipped_budget'
            else:
                timeout = max(1, int(min(step.timeout, remaining)))
                t0 = time.monotonic()
                try:
                    p = subprocess.run(step.argv, cwd=source, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False,
                                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
                    log = clean_log(p.stdout)
                    entry['status'] = 'passed' if p.returncode == 0 else 'failed'
                    entry['returncode'] = p.returncode
                except subprocess.TimeoutExpired as e:
                    entry['status'] = 'timeout'
                    raw = e.stdout or b''
                    log = clean_log(raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else raw)
                except OSError as e:
                    entry['status'] = 'unavailable'
                    log = clean_log(str(e))
                entry['elapsed_seconds'] = round(time.monotonic() - t0, 2)
                # Report contains only bounded/redacted output; do not upload raw logs.
                entry['log'] = log
                entry['log_sha256'] = hashlib.sha256(log.encode('utf-8')).hexdigest()
        result['steps'].append(entry)
        atomic_json(output, result)  # preserve completed progress after each step
    statuses = [x['status'] for x in result['steps']]
    result['status'] = ('planned' if not execute else
                        'passed' if all(x == 'passed' for x in statuses) else 'incomplete')
    result['completed_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(output, result)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', choices=PROJECTS, default='studio')
    p.add_argument('--mode', choices=['audit', 'test'], default='audit')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true', help='Run only preapproved recipes')
    p.add_argument('--budget-seconds', type=int, default=600)
    args = p.parse_args()
    outcome = run(args.project, args.mode, args.source, args.output,
                  execute=args.execute, budget_seconds=args.budget_seconds)
    print(json.dumps({k: outcome[k] for k in ('project', 'mode', 'status', 'execute')}, ensure_ascii=False))
    return 0 if outcome['status'] in {'passed', 'planned'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
