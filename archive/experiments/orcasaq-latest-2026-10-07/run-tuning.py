#!/usr/bin/env python3
"""Bounded latest-engine parameter sweep; preserves production and all prior evidence."""
import concurrent.futures
import copy
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('qualification', HERE / 'run-qualification.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
w, q = mod.w, mod.q
ROOT = HERE / 'tuning'
BASE = HERE / 'qualification'
IMAGE = 'sha256:bf0c4eebc48d0db6d8c779be8b81018ced33acfbcc1ac98011ad9ac82c4ee45c'
CONFIGS = [('control-before', 2, 2048), ('mtp1-b2048', 1, 2048),
           ('mtp3-b2048', 3, 2048), ('mtp0-b2048', 0, 2048),
           ('mtp2-b4096', 2, 4096), ('mtp3-b4096', 3, 4096),
           ('control-after', 2, 2048)]

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]

def stop(thread):
    errors = []
    if w.active:
        try:
            with (w.ROOT / 'server.log').open('w') as f:
                subprocess.run(['docker', 'logs', w.active], stdout=f, stderr=f, timeout=60)
        except Exception as exc:
            errors.append(str(exc))
        finally:
            w.command(['docker', 'rm', '-f', w.active], 90)
            w.active = None
    w.stop_monitor.set()
    if thread:
        thread.join(timeout=30)
    if errors:
        (w.ROOT / 'log-capture-errors.json').write_text(json.dumps(errors))

def launch(name, depth, batch):
    w.ROOT = ROOT / name
    w.ROOT.mkdir(parents=True)
    w.active = 'orcasaq-tuning'
    args = ['docker', 'run', '-d', '--name', w.active, '--gpus', 'all', '--network', 'host',
            '--memory', '32g', '--memory-swap', '32g', '--cpus', '8', '--shm-size', '8g',
            '-v', '/data/models/qwen3.8/orcarouter/OrcaSAQ2-27B:/models/orca:ro',
            '-v', '/tmp/orcasaq-latest-tuning-cache:/cache']
    extra = '--speculative-config ' + json.dumps({'method': 'qwen3_next_mtp', 'num_speculative_tokens': depth}, separators=(',', ':')) if depth else ''
    for item in ['PRESET=16gb-mtp', 'MODEL=/models/orca', 'SERVED_NAME=orca-qwen3.8-27b',
                 'PORT=18185', 'BIND=127.0.0.1', 'MAXLEN=150000', 'MAXSEQS=4',
                 f'MAXBATCHTOK={batch}', f'EXTRA={extra}', 'XDG_CACHE_HOME=/cache',
                 'TORCH_EXTENSIONS_DIR=/cache/torch_extensions']:
        args += ['-e', item]
    args.append(IMAGE)
    w.save('launch.json', args)
    w.save('settings.json', {'mtp_depth': depth, 'batch_tokens': batch, 'sequence_slots': 4, 'image': IMAGE})
    w.command(args)
    w.stop_monitor.clear(); w.abort.clear()
    thread = threading.Thread(target=w.monitor, args=(w.active, w.ROOT / 'telemetry.jsonl'))
    thread.start()
    return thread

def metrics(arm):
    try:
        import urllib.request
        with urllib.request.urlopen(arm['base_url'].removesuffix('/v1') + '/metrics', timeout=10) as r:
            return r.read().decode()
    except Exception as exc:
        return '# metrics unavailable: ' + str(exc)

def screen(arm):
    plan = copy.deepcopy(w.h.load_plan(BASE / 'plan-speed-repeat.json'))
    plan.pop('sha256')
    plan['arms'] = [copy.deepcopy(arm)]
    plan['arms'][0]['metadata']['launch'] = json.loads((w.ROOT / 'settings.json').read_text())
    plan['sha256'] = w.h.digest(plan)
    w.save('plan-speed.json', plan)
    w.h.run(plan, arm['name'], w.ROOT / 'speed', 1800)
    w.save('speed-summary.json', w.h.summarize(plan, w.ROOT / 'speed'))
    w.CONCURRENCY_CASES = [(2, 0, False), (2, 700, False)]
    w.concurrency(arm)
    (w.ROOT / 'metrics-after-screen.txt').write_text(metrics(arm))
    bs = rows(w.ROOT / 'orca_vllm-concurrency.jsonl')
    # Fixed output budgets make this speed screen independent of code answer length.
    elapsed = sum(b['batch_wall_s'] for b in bs)
    okay = all(r['result']['status'] == 'ok' and r['result']['usage']['completion_tokens'] == 512 for b in bs for r in b['rows'])
    w.save('screen.json', {'screen_seconds': elapsed, 'fixed_output_complete': okay})
    return {'name': w.ROOT.name, 'screen_seconds': elapsed, 'fixed_output_complete': okay}

def quality_panel(arm, name, depth, batch):
    thread = launch(name, depth, batch)
    try:
        w.wait_ready()
        q.run(w, arm, w.ROOT / 'orca_vllm-concurrent-quality.jsonl')
        bs = rows(w.ROOT / 'orca_vllm-concurrent-quality.jsonl')
        rs = [r for b in bs for r in b['rows']]
        result = {'passed': sum(r['score'] == 1 for r in rs), 'count': len(rs),
                  'inference_seconds': sum(b['inference_wall_s'] for b in bs),
                  'failures': [r['id'] for r in rs if r['score'] != 1]}
        w.save('panel-summary.json', result)
        return result
    finally:
        stop(thread)

def main():
    ROOT.mkdir(exist_ok=True)
    if (ROOT / 'state-before.json').exists():
        raise RuntimeError('Refusing to overwrite a started campaign')
    if not json.loads((BASE / 'restored-public-verification.json').read_text())['passed']:
        raise RuntimeError('Previous restoration was not verified')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, w.interrupted)
    for service in ('vram-thermal-guard.service', 'vram-fan-curve.service'):
        w.command(['systemctl', 'is-active', '--quiet', service])
    if Path('/run/gpu-thermal-trip').exists():
        raise RuntimeError('Thermal trip latched')
    Path('/tmp/orcasaq-latest-tuning-cache').mkdir(exist_ok=True)
    replicas = w.command(['kubectl', '-n', 'apps', 'get', 'deployment', 'llama', '-o', 'jsonpath={.spec.replicas}'])
    suspended = w.command(['kubectl', '-n', 'flux-system', 'get', 'kustomization', 'apps', '-o', 'jsonpath={.spec.suspend}']) == 'true'
    template = w.command(['kubectl', '-n', 'apps', 'get', 'deployment', 'llama', '-o', 'jsonpath={.spec.template}'])
    w.ROOT = ROOT
    w.save('state-before.json', {'replicas': replicas, 'flux_suspended': suspended, 'template_sha256': w.h.digest(template)})
    arm = next(a for a in w.h.load_plan(BASE / 'plan-full.json')['arms'] if a['name'] == 'orca_vllm')
    arm['metadata'].update(runtime='vLLM0.31.0 + ExLlamaV3 1.5.4 + Orca plugin7ddffb12', image=IMAGE,
                           cache_policy='fresh server with shared compilation cache only; one excluded harness warmup; prefix cache enabled')
    thread = None
    try:
        if not suspended:
            w.command(['flux', 'suspend', 'kustomization', 'apps'])
        w.command(['kubectl', '-n', 'apps', 'scale', 'deployment', 'llama', '--replicas=0'])
        w.command(['kubectl', '-n', 'apps', 'wait', '--for=delete', 'pod', '-l', 'app.kubernetes.io/name=llama', '--timeout=10m'], 660)
        if w.command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']):
            raise RuntimeError('GPU remains occupied')
        screens = []
        for name, depth, batch in CONFIGS:
            print(f'SCREEN {name}: MTP={depth} batch={batch}', flush=True)
            thread = launch(name, depth, batch)
            try:
                w.wait_ready()
                screens.append(screen(arm))
                print('SCREEN RESULT ' + json.dumps(screens[-1]), flush=True)
            finally:
                stop(thread); thread = None
        w.ROOT = ROOT
        w.save('screens.json', screens)
        controls = [s for s in screens if s['name'].startswith('control')]
        drift = abs(controls[1]['screen_seconds'] / controls[0]['screen_seconds'] - 1)
        candidates = sorted((s for s in screens if not s['name'].startswith('control') and s['fixed_output_complete']), key=lambda s: s['screen_seconds'])[:2]
        panels = []
        for candidate in candidates:
            _, depth, batch = next(c for c in CONFIGS if c[0] == candidate['name'])
            result = quality_panel(arm, 'panel-' + candidate['name'], depth, batch)
            panels.append({**candidate, 'panel': result, 'mtp_depth': depth, 'batch_tokens': batch})
        # Prefer the original publisher preset unless a screened candidate passes all
        # paired coding cases and beats both controls by at least five percent.
        gate = min(s['screen_seconds'] for s in controls) * .95
        qualified = [p for p in panels if p['panel']['passed'] == p['panel']['count'] == 24 and p['screen_seconds'] <= gate and drift <= .05]
        selected = min(qualified, key=lambda p: p['screen_seconds']) if qualified else {'name': 'publisher-control', 'mtp_depth': 2, 'batch_tokens': 2048}
        w.ROOT = ROOT
        w.save('selection.json', {'control_drift_fraction': drift, 'screen_gate_seconds': gate,
                                  'panels': panels, 'selected': selected,
                                  'scope': 'Provisional screen selection; full suite and long-context validation follow.'})
        print('SELECTED ' + json.dumps(selected), flush=True)
        thread = launch('selected-validation', selected['mtp_depth'], selected['batch_tokens'])
        w.wait_ready()
        plan = copy.deepcopy(w.h.load_plan(BASE / 'plan-full.json'))
        plan.pop('sha256')
        plan['arms'] = [copy.deepcopy(arm)]
        plan['arms'][0]['metadata']['launch'] = selected
        plan['sha256'] = w.h.digest(plan)
        w.save('plan-full.json', plan)
        w.h.run(plan, arm['name'], w.ROOT / 'full', 1800)
        w.save('full-summary.json', w.h.summarize(plan, w.ROOT / 'full'))
        w.CONCURRENCY_CASES = [(2, 5200, True)]
        w.concurrency(arm)
        (w.ROOT / 'metrics-final.txt').write_text(metrics(arm))
    finally:
        cleanup = []
        try:
            stop(thread)
        except Exception as exc:
            cleanup.append('candidate cleanup: ' + str(exc))
        for args in ([['kubectl', '-n', 'apps', 'scale', 'deployment', 'llama', '--replicas=' + replicas]] + ([] if suspended else [['flux', 'resume', 'kustomization', 'apps']])):
            try:
                w.command(args)
            except Exception as exc:
                cleanup.append(str(exc))
        try:
            w.command(['kubectl', '-n', 'apps', 'rollout', 'status', 'deployment/llama', '--timeout=15m'], 960)
            actual = w.command(['kubectl', '-n', 'apps', 'get', 'deployment', 'llama', '-o', 'jsonpath={.spec.template}'])
            if w.h.digest(actual) != w.h.digest(template):
                cleanup.append('Deployment template changed')
        except Exception as exc:
            cleanup.append(str(exc))
        w.ROOT = ROOT
        w.save('state-restored.json', {'restored': not cleanup, 'failures': cleanup})
        print('RESTORED ' + str(not cleanup), flush=True)
        if cleanup:
            raise RuntimeError('Restoration requires attention')

if __name__ == '__main__':
    main()
