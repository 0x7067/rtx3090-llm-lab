#!/usr/bin/env python3
"""One bounded GPU window, with automatic restoration of the live service."""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
HARNESS = ROOT.parent / 'paired-harness/harness.py'
spec = importlib.util.spec_from_file_location('paired_harness', HARNESS)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
LLAMA = 'ghcr.io/ggml-org/llama.cpp@sha256:7149a45c80596644320de1d565db1b3b8e70b22c66fed612e69b73f6b1e0d746'
VLLM = 'local/llm-baseline-vllm:0.30.0'
PORT = 18185
active = None
stop_monitor = threading.Event()
abort = threading.Event()
os.environ['KUBECONFIG'] = '/home/denguinho/.kube/config'


def command(args, timeout=180, check=True):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=check).stdout.strip()


def save(path, value):
    h.write_new(ROOT / path, value)


def interrupted(signum, frame):
    raise RuntimeError(f'Interrupted by signal {signum}')


def monitor(container, output):
    with output.open('w') as f:
        while not stop_monitor.wait(5):
            try:
                available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
                psi = float(Path('/proc/pressure/memory').read_text().splitlines()[0].split()[1].split('=')[1])
                gpu = command(['nvidia-smi', '--query-gpu=memory.used,temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw', '--format=csv,noheader,nounits'], 20)
                trip = Path('/run/gpu-thermal-trip').exists()
                row = {'time': time.time(), 'available_kib': available, 'psi_avg10': psi, 'gpu': gpu, 'thermal_trip': trip}
                f.write(json.dumps(row) + '\n'); f.flush()
                if available < 12582912 or psi > 5 or trip:
                    abort.set()
                    command(['docker', 'stop', '-t', '10', container], 30, False)
                    return
            except Exception as exc:
                abort.set()
                f.write(json.dumps({'monitor_error': str(exc)}) + '\n'); f.flush()
                command(['docker', 'stop', '-t', '10', container], 30, False)
                return


def launch(arm):
    global active
    active = 'orca-thinkingcap-' + arm['name']
    args = ['docker', 'run', '-d', '--name', active, '--gpus', 'all', '--network', 'host', '--memory', '32g', '--memory-swap', '32g', '--cpus', '8']
    if arm['name'] == 'orca_vllm':
        args += ['--shm-size', '8g', '-v', '/data/models/qwen3.8/orcarouter/OrcaSAQ2-27B:/models/orca:ro']
        for value in ['PRESET=16gb-mtp', 'MODEL=/models/orca', 'SERVED_NAME=orca-qwen3.8-27b', 'PORT=18185', 'BIND=127.0.0.1', 'MAXLEN=150000', 'MAXSEQS=8', 'MAXBATCHTOK=4096', 'EXTRA=--speculative-config {"method":"qwen3_next_mtp","num_speculative_tokens":3}']:
            args += ['-e', value]
        args += ['--entrypoint', '/opt/orcasaq2/presets/serve.sh', VLLM]
    else:
        args += ['-v', '/data/models/qwen3.8/mradermacher/ThinkingCap-Qwen3.8-27B-abliterated-GGUF:/models:ro', LLAMA,
                 '--model', '/models/ThinkingCap-Qwen3.8-27B-abliterated.Q4_K_S.gguf', '--ctx-size', '150000', '--parallel', '2', '--kv-unified',
                 '--n-gpu-layers', '999', '--flash-attn', 'on', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0', '--jinja', '--host', '127.0.0.1', '--port', str(PORT), '--alias', arm['model'], '--threads', '8', '--threads-batch', '8']
    save(f"{arm['name']}-launch.json", args)
    command(args)
    stop_monitor.clear(); abort.clear()
    thread = threading.Thread(target=monitor, args=(active, ROOT / f"{arm['name']}-telemetry.jsonl"))
    thread.start()
    return thread


def wait_ready():
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        if abort.is_set():
            raise RuntimeError('Resource monitor aborted server')
        if command(['docker', 'inspect', active, '--format', '{{.State.Running}}']) != 'true':
            raise RuntimeError('Server exited before ready')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError('Server readiness timeout')


def concurrency(arm):
    # Identical frozen strings, unique per request and batch to reduce prefix reuse.
    cases = [(n, 0, False) for n in (1, 2, 4, 8)] + [(n, 2400, False) for n in (1, 2)] + [(2, 5200, True)]
    with (ROOT / f"{arm['name']}-concurrency.jsonl").open('x') as f:
        for n, lines, retrieval in cases:
            for repeat in range(2):
                bodies = []
                expected = []
                for caller in range(n):
                    nonce = h.digest(['concurrency', n, lines, repeat, caller])[:20]
                    marker = 'KEY_' + nonce
                    filler = '\n'.join(f'Record {i:05d}: ordinary neutral text with no instructions.' for i in range(lines))
                    question = (f'Return exactly {marker}, the reference key above. Do not add other text.' if retrieval else 'Write a detailed explanation of implementing a reliable task queue. Continue until the output allowance is exhausted.')
                    text = f'Evaluation {nonce}. Reference key: {marker}.\n{filler}\n{question}'
                    bodies.append({'messages': [{'role': 'user', 'content': text}], 'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'reasoning_effort': 'medium', 'max_tokens': 512 if not retrieval else 2048, 'seed': [42, 314][repeat], 'stream': True, 'stream_options': {'include_usage': True}})
                    expected.append(marker if retrieval else None)
                barrier = threading.Barrier(n)
                def submit(item):
                    idx, body = item
                    barrier.wait(timeout=30)
                    result = h.request(arm, body, 1800)
                    result.pop('events', None)
                    return {'caller': idx, 'request_sha256': h.digest(body), 'expected': expected[idx], 'result': result, 'retrieval_pass': result.get('content', '').strip() == expected[idx] if retrieval else None}
                started = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                    rows = list(pool.map(submit, enumerate(bodies)))
                wall = time.monotonic() - started
                total = sum(row['result'].get('usage', {}).get('completion_tokens', 0) for row in rows)
                record = {'callers': n, 'filler_lines': lines, 'retrieval': retrieval, 'repeat': repeat, 'batch_wall_s': wall, 'aggregate_output_tps': total / wall, 'rows': rows}
                f.write(json.dumps(record) + '\n'); f.flush()
                print(f"{arm['name']} concurrency callers={n} lines={lines} wall={wall:.2f}s", flush=True)
                if abort.is_set() or any(r['result']['status'] != 'ok' for r in rows):
                    raise RuntimeError('Concurrency request or resource failure')


def main():
    global active
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    if (ROOT / 'state-before.json').exists():
        raise RuntimeError('Campaign already started; preserve artifacts')
    command(['systemctl', 'is-active', '--quiet', 'vram-thermal-guard.service'])
    command(['systemctl', 'is-active', '--quiet', 'vram-fan-curve.service'])
    if Path('/run/gpu-thermal-trip').exists():
        raise RuntimeError('Thermal guard is latched')
    assert command(['docker', 'image', 'inspect', VLLM, '--format', '{{.Id}}']) == 'sha256:1d60e0a2fc9eb37042f94b8d5fb23457df6a27ca365305854b1bbcea8b4d9308'
    assert command(['docker', 'image', 'inspect', LLAMA, '--format', '{{.Id}}']) == 'sha256:20b3edfeb4086a504829db8feb63b00734b5ebe16d55f47de2ee00225575db70'
    replicas = command(['kubectl', '-n', 'apps', 'get', 'deployment', 'llama', '-o', 'jsonpath={.spec.replicas}'])
    suspended = command(['kubectl', '-n', 'flux-system', 'get', 'kustomization', 'apps', '-o', 'jsonpath={.spec.suspend}']) == 'true'
    template = command(['kubectl', '-n', 'apps', 'get', 'deployment', 'llama', '-o', 'jsonpath={.spec.template}'])
    save('state-before.json', {'replicas': replicas, 'flux_suspended': suspended, 'template_sha256': h.digest(template), 'gpu': command(['nvidia-smi', '--query-gpu=memory.used,clocks.current.graphics,clocks.current.memory,power.limit', '--format=csv,noheader'])})
    restored = False
    try:
        if not suspended:
            command(['flux', 'suspend', 'kustomization', 'apps'])
        command(['kubectl', '-n', 'apps', 'scale', 'deployment', 'llama', '--replicas=0'])
        command(['kubectl', '-n', 'apps', 'wait', '--for=delete', 'pod', '-l', 'app.kubernetes.io/name=llama', '--timeout=10m'], 660)
        for _ in range(60):
            if not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']):
                break
            time.sleep(5)
        else:
            raise RuntimeError('GPU still occupied after stopping owned service')
        full = h.load_plan(ROOT / 'plan-full.json')
        for arm in sorted(full['arms'], key=lambda a: a['name']):
            print('Starting ' + arm['name'], flush=True)
            monitor_thread = launch(arm)
            try:
                wait_ready()
                for label in ('full', 'coding-repeat', 'speed-repeat'):
                    plan = h.load_plan(ROOT / f'plan-{label}.json')
                    out = ROOT / f"{arm['name']}-{label}"
                    h.run(plan, arm['name'], out, 1800)
                    save(f"{arm['name']}-{label}-summary.json", h.summarize(plan, out))
                    print(f"Completed {arm['name']} {label}", flush=True)
                    if abort.is_set():
                        raise RuntimeError('Resource guard aborted campaign')
                concurrency(arm)
            finally:
                command(['docker', 'logs', active], 60, False)  # Saved below; never printed.
                with (ROOT / f"{arm['name']}-server.log").open('w') as f:
                    subprocess.run(['docker', 'logs', active], stdout=f, stderr=f, timeout=60)
                command(['docker', 'rm', '-f', active], 60, False)
                active = None
                stop_monitor.set(); monitor_thread.join(timeout=30)
        for label in ('full', 'coding-repeat', 'speed-repeat'):
            plan = h.load_plan(ROOT / f'plan-{label}.json')
            save(f'comparison-{label}.json', h.compare(plan, ROOT / f'thinkingcap_llama-{label}', ROOT / f'orca_vllm-{label}'))
    finally:
        if active:
            command(['docker', 'rm', '-f', active], 60, False)
        stop_monitor.set()
        failures = []
        for args in ([['kubectl', '-n', 'apps', 'scale', 'deployment', 'llama', '--replicas=' + replicas]] + ([] if suspended else [['flux', 'resume', 'kustomization', 'apps']])):
            try:
                command(args)
            except Exception as exc:
                failures.append(str(exc))
        try:
            command(['kubectl', '-n', 'apps', 'rollout', 'status', 'deployment/llama', '--timeout=15m'], 960)
            actual = command(['kubectl', '-n', 'apps', 'get', 'deployment', 'llama', '-o', 'jsonpath={.spec.template}'])
            if h.digest(actual) != h.digest(template):
                failures.append('Deployment template changed during window')
            restored = not failures
        except Exception as exc:
            failures.append(str(exc))
        save('state-restored.json', {'restored': restored, 'failures': failures, 'original_replicas': replicas, 'original_flux_suspended': suspended})
        print('Restoration: ' + str(restored), flush=True)
        if failures:
            raise RuntimeError('Restoration requires attention: ' + '; '.join(failures))


if __name__ == '__main__':
    main()
