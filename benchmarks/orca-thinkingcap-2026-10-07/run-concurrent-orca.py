#!/usr/bin/env python3
"""A short, separately recorded Orca window for the two-agent coding panel."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess

SOURCE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('window', SOURCE / 'run.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
w.ROOT = SOURCE / 'orca-concurrent-quality'
w.ROOT.mkdir(exist_ok=True)
qs = importlib.util.spec_from_file_location('concurrent_quality', SOURCE / 'concurrent_quality.py')
q = importlib.util.module_from_spec(qs)
qs.loader.exec_module(q)


def main():
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, w.interrupted)
    if (w.ROOT / 'state-before.json').exists():
        raise RuntimeError('Refusing to overwrite an executed window')
    if not json.loads((SOURCE / 'thinkingcap-200k/state-restored.json').read_text())['restored']:
        raise RuntimeError('Previous service restoration incomplete')
    for service in ('vram-thermal-guard.service', 'vram-fan-curve.service'):
        w.command(['systemctl', 'is-active', '--quiet', service])
    if Path('/run/gpu-thermal-trip').exists():
        raise RuntimeError('Thermal trip latched')
    replicas = w.command(['kubectl', '-n', 'apps', 'get', 'deployment', 'llama', '-o', 'jsonpath={.spec.replicas}'])
    suspended = w.command(['kubectl', '-n', 'flux-system', 'get', 'kustomization', 'apps', '-o', 'jsonpath={.spec.suspend}']) == 'true'
    template = w.command(['kubectl', '-n', 'apps', 'get', 'deployment', 'llama', '-o', 'jsonpath={.spec.template}'])
    w.save('state-before.json', {'replicas': replicas, 'flux_suspended': suspended, 'template_sha256': w.h.digest(template)})
    thread = None
    try:
        if not suspended:
            w.command(['flux', 'suspend', 'kustomization', 'apps'])
        w.command(['kubectl', '-n', 'apps', 'scale', 'deployment', 'llama', '--replicas=0'])
        w.command(['kubectl', '-n', 'apps', 'wait', '--for=delete', 'pod', '-l', 'app.kubernetes.io/name=llama', '--timeout=10m'], 660)
        if w.command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']):
            raise RuntimeError('GPU remains occupied')
        arm = next(a for a in w.h.load_plan(SOURCE / 'plan-full.json')['arms'] if a['name'] == 'orca_vllm')
        thread = w.launch(arm)
        w.wait_ready()
        q.run(w, arm, w.ROOT / 'orca_vllm-concurrent-quality.jsonl')
        w.CONCURRENCY_CASES = [(2, 5200, True)]
        w.concurrency(arm)
    finally:
        if w.active:
            with (w.ROOT / 'server.log').open('w') as f:
                subprocess.run(['docker', 'logs', w.active], stdout=f, stderr=f, timeout=60)
            w.command(['docker', 'rm', '-f', w.active], 60, False)
        w.stop_monitor.set()
        if thread:
            thread.join(timeout=30)
        failures = []
        for args in ([['kubectl', '-n', 'apps', 'scale', 'deployment', 'llama', '--replicas=' + replicas]] + ([] if suspended else [['flux', 'resume', 'kustomization', 'apps']])):
            try:
                w.command(args)
            except Exception as exc:
                failures.append(str(exc))
        try:
            w.command(['kubectl', '-n', 'apps', 'rollout', 'status', 'deployment/llama', '--timeout=15m'], 960)
            actual = w.command(['kubectl', '-n', 'apps', 'get', 'deployment', 'llama', '-o', 'jsonpath={.spec.template}'])
            if w.h.digest(actual) != w.h.digest(template):
                failures.append('Deployment template changed')
        except Exception as exc:
            failures.append(str(exc))
        w.save('state-restored.json', {'restored': not failures, 'failures': failures})
        print('Orca concurrent-quality restoration: ' + str(not failures), flush=True)
        if failures:
            raise RuntimeError('Restoration requires attention')


if __name__ == '__main__':
    main()
