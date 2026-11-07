"""Optional advisory review using a user-selected OpenAI-compatible endpoint.

Only the bounded redacted audit report is sent, and only after explicit --send.
The model never receives source files, writes commits, places orders or posts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def suggest(report: dict, *, base_url: str, model: str, api_key: str = '', send: bool = False) -> dict:
    if not send:
        return {'status': 'disabled', 'reason': 'AI calls require --send'}
    if not isinstance(report, dict) or report.get('schema') != 1:
        raise ValueError('Invalid wolf-ai report')
    parts = urlsplit(base_url)
    if parts.scheme not in {'https', 'http'} or not parts.netloc:
        raise ValueError('Invalid endpoint')
    # Non-loopback HTTP could expose code/information and credentials.
    if parts.scheme == 'http' and parts.hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('Insecure remote HTTP is prohibited')
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('Do not embed credentials or query parameters in endpoint')
    url = base_url.rstrip('/') + '/chat/completions'
    # Prevent sending raw code or arbitrary text-heavy/secret-bearing fields.
    brief = {'project': report.get('project'), 'mode': report.get('mode'),
             'status': report.get('status'),
             'steps': [{'name': x.get('name'), 'status': x.get('status'),
                        'returncode': x.get('returncode')}
                       for x in report.get('steps', [])[:12]]}
    body = {'model': model, 'max_tokens': 700, 'temperature': 0,
            'messages': [
                {'role': 'system', 'content': 'You are a careful engineering auditor. Give up to five concrete next actions. Do not claim an issue is fixed or a test passed without evidence. No code or secrets are provided.'},
                {'role': 'user', 'content': json.dumps(brief, ensure_ascii=False)}]}
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    request = Request(url, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    message = payload['choices'][0]['message']['content']
    return {'status': 'advisory_only', 'model': model, 'advice': str(message)[:5000],
            'no_source_transmitted': True, 'no_code_changes': True}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--send', action='store_true')
    p.add_argument('--base-url', default=os.getenv('WOLF_LLM_BASE_URL', 'http://127.0.0.1:11434/v1'))
    p.add_argument('--model', default=os.getenv('WOLF_LLM_MODEL', 'qwen2.5-coder:7b'))
    args = p.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    result = suggest(report, base_url=args.base_url, model=args.model,
                     api_key=os.getenv('WOLF_LLM_API_KEY', ''), send=args.send)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(result['status'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
