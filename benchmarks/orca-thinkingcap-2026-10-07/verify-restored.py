#!/usr/bin/env python3
"""Verify the restored route with a credential kept only in process memory."""
import base64
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
BASE = 'https://llama.denguinho.org/v1'
os.environ['KUBECONFIG'] = '/home/denguinho/.kube/config'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def kube(args):
    return json.loads(subprocess.run(['kubectl', '-n', 'apps', *args, '-o', 'json'], check=True, capture_output=True, text=True, timeout=90).stdout)


def main():
    path = ROOT / 'restored-public-verification.json'
    if path.exists():
        raise RuntimeError('Refusing to overwrite verification evidence')
    deployment = kube(['get', 'deployment', 'llama'])
    env = [e for c in deployment['spec']['template']['spec']['containers'] for e in c.get('env', []) if e['name'] == 'LLAMA_API_TOKEN']
    ref = env[0]['valueFrom']['secretKeyRef']
    secret = kube(['get', 'secret', ref['name']])
    token = base64.b64decode(secret['data'][ref['key']]).decode()
    opener = urllib.request.build_opener(NoRedirect())
    report = {'ready_replicas': deployment['status'].get('readyReplicas', 0)}

    def request(path, body=None, authorized=False):
        headers = {'User-Agent': 'home-llm-benchmark-verification/1.0'}
        if authorized:
            headers['Authorization'] = 'Bearer ' + token
        if body is not None:
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
        try:
            with opener.open(req, timeout=900) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, None

    report['unauthenticated_status'], _ = request('/models')
    report['authenticated_status'], catalog = request('/models', authorized=True)
    report['model_ids'] = [m['id'] for m in (catalog or {}).get('data', [])]
    if report['authenticated_status'] == 200:
        code, response = request('/chat/completions', {'model': 'orca-saq2-27b', 'messages': [{'role': 'user', 'content': 'Reply exactly BENCHMARK_RESTORED.'}], 'temperature': 0, 'reasoning_effort': 'low', 'max_tokens': 256}, authorized=True)
        report['completion_status'] = code
        content = (response or {}).get('choices', [{}])[0].get('message', {}).get('content') or ''
        report['completion_exact'] = content.strip() == 'BENCHMARK_RESTORED'
    report['passed'] = report['ready_replicas'] == 1 and report['unauthenticated_status'] == 401 and report['authenticated_status'] == 200 and report.get('completion_exact', False)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    if not report['passed']:
        raise RuntimeError('Restored public verification did not pass')


if __name__ == '__main__':
    main()
