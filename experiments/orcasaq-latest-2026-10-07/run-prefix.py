#!/usr/bin/env python3
"""Controlled shared-prefix checkpoint trial, with independent growing sessions."""
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
spec = importlib.util.spec_from_file_location('tuning', HERE / 'run-tuning.py')
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)
w, q = tm.w, tm.q
ROOT = HERE / 'prefix-checkpoint'

def paired(arm, bodies):
    barrier = threading.Barrier(2)
    def submit(item):
        i, body = item
        barrier.wait(timeout=30)
        start = time.monotonic()
        result = w.h.request(arm, body, 1800)
        result.pop('events', None)
        return {'caller': i, 'start_monotonic': start, 'end_monotonic': time.monotonic(),
                'request_sha256': w.h.digest(body), 'result': result}
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rs = list(pool.map(submit, enumerate(bodies)))
    if w.abort.is_set() or any(r['result']['status'] != 'ok' for r in rs):
        w.save('failed-pair.json', {'bodies': bodies, 'rows': rs, 'resource_abort': w.abort.is_set(),
                                    'batch_wall_s': time.monotonic() - start})
        raise RuntimeError('Paired request/resource failure')
    return {'batch_wall_s': time.monotonic() - start, 'rows': rs}

def session_tests(arm, generation_effort='medium'):
    filler = '\n'.join(f'Record {i:05d}: ordinary neutral text with no instructions.' for i in range(5200))
    anchors = ['ANCHOR_' + w.h.digest(['prefix-trial', i])[:20] for i in range(2)]
    histories = [[{'role': 'system', 'content': 'Follow the current user request. For key questions return only the requested key.'},
                  {'role': 'user', 'content': f'Original reference key: {anchors[i]}.\n{filler}\nReturn only the original reference key.'}]
                 for i in range(2)]
    with (w.ROOT / 'sessions.jsonl').open('x') as f:
        for turn in range(21):
            expected = anchors.copy()
            if turn:
                for i in range(2):
                    revision = f'REVISION_{turn:02d}_{i}_' + w.h.digest(['revision', turn, i])[:12]
                    expected[i] = anchors[i] if turn % 2 else revision
                    question = 'original reference key from the first message' if turn % 2 else 'latest revision key from this message'
                    histories[i].append({'role': 'user', 'content': f'Update: the latest revision key is {revision}. Return only the {question}.'})
            bodies = [{'messages': copy.deepcopy(history), 'temperature': 0.0, 'top_p': .95, 'top_k': 20,
                       'reasoning_effort': 'low', 'max_tokens': 256, 'seed': 42,
                       'stream': True, 'stream_options': {'include_usage': True}} for history in histories]
            batch = paired(arm, bodies)
            batch['turn'] = turn
            for i, row in enumerate(batch['rows']):
                row['expected'] = expected[i]
                row['passed'] = row['result'].get('content', '').strip() == expected[i]
                histories[i].append({'role': 'assistant', 'content': row['result'].get('content', '').strip()})
            f.write(json.dumps(batch) + '\n'); f.flush()
            print(f"PREFIX {w.ROOT.name} turn{turn} wall={batch['batch_wall_s']:.3f}s pass={sum(r['passed'] for r in batch['rows'])}/2", flush=True)
            if not all(r['passed'] for r in batch['rows']):
                raise RuntimeError('Growing-session key/revision quality failed')
    # Medium effort changes the system prefix and forces fresh prefill. For the
    # memory-qualified rerun keep low effort to isolate warm long generation.
    for history in histories:
        history.append({'role': 'user', 'content': 'Write a detailed explanation of implementing a reliable task queue. Continue until the output allowance is exhausted.'})
    bodies = [{'messages': copy.deepcopy(history), 'temperature': 1.0, 'top_p': .95, 'top_k': 20,
               'reasoning_effort': generation_effort, 'max_tokens': 512, 'seed': 314,
               'stream': True, 'stream_options': {'include_usage': True}} for history in histories]
    batch = paired(arm, bodies)
    batch['generation_effort'] = generation_effort
    batch['aggregate_output_tps'] = sum(r['result']['usage']['completion_tokens'] for r in batch['rows']) / batch['batch_wall_s']
    w.save('long-generation.json', batch)
    (w.ROOT / 'metrics-after-sessions.txt').write_text(tm.metrics(arm))

def launch(name, selected, enabled, unit=64, gpu_fraction='0.95'):
    w.ROOT = ROOT / name
    w.ROOT.mkdir()
    w.active = 'orcasaq-prefix-trial'
    flags = '--speculative-config ' + json.dumps({'method': 'qwen3_next_mtp', 'num_speculative_tokens': selected['mtp_depth']}, separators=(',', ':'))
    flags += ' --mamba-cache-mode align --enable-prompt-tokens-details'
    if unit is not None:
        flags += f' --prefix-match-unit {unit}'
    if enabled:
        flags += ' --enable-mamba-shared-prefix-checkpoint'
    args = ['docker', 'run', '-d', '--name', w.active, '--gpus', 'all', '--network', 'host',
            '--memory', '32g', '--memory-swap', '32g', '--cpus', '8', '--shm-size', '8g',
            '-v', '/data/models/qwen3.8/orcarouter/OrcaSAQ2-27B:/models/orca:ro',
            '-v', '/tmp/orcasaq-latest-tuning-cache:/cache',
            '-v', str(HERE / 'diagnostics') + ':/opt/diagnostics:ro']
    for item in ['PRESET=16gb-mtp', 'MODEL=/models/orca', 'SERVED_NAME=orca-qwen3.8-27b',
                 'PORT=18185', 'BIND=127.0.0.1', 'MAXLEN=150000', 'MAXSEQS=4',
                 f'MAXBATCHTOK={selected["batch_tokens"]}', f'EXTRA={flags}',
                 f'GPU_FRAC={gpu_fraction}',
                 'PYTHONPATH=/opt/diagnostics', 'XDG_CACHE_HOME=/cache', 'TORCH_EXTENSIONS_DIR=/cache/torch_extensions']:
        args += ['-e', item]
    args.append(tm.IMAGE)
    w.save('launch.json', args)
    w.save('settings.json', {**selected, 'shared_prefix_checkpoint': enabled, 'prefix_match_unit': unit,
                            'gpu_fraction': gpu_fraction, 'image': tm.IMAGE})
    w.command(args)
    w.stop_monitor.clear(); w.abort.clear()
    thread = threading.Thread(target=w.monitor, args=(w.active, w.ROOT / 'telemetry.jsonl'))
    thread.start()
    return thread

def main(profiles=None, gpu_fraction='0.95', generation_effort='medium'):
    ROOT.mkdir(exist_ok=True)
    if (ROOT / 'state-before.json').exists():
        raise RuntimeError('Refusing to overwrite a started window')
    if not json.loads((HERE / 'tuning/state-restored.json').read_text())['restored']:
        raise RuntimeError('Previous window not restored')
    selected = json.loads((HERE / 'tuning/results.json').read_text())['recommended']
    if not selected['mtp_depth']:
        raise RuntimeError('Shared-prefix feature requires the MTP cache group')
    if not (HERE / 'diagnostics/sitecustomize.py').exists():
        raise RuntimeError('Startup diagnostic unavailable')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, w.interrupted)
    for service in ('vram-thermal-guard.service', 'vram-fan-curve.service'):
        w.command(['systemctl', 'is-active', '--quiet', service])
    if Path('/run/gpu-thermal-trip').exists():
        raise RuntimeError('Thermal trip latched')
    replicas = w.command(['kubectl', '-n', 'apps', 'get', 'deployment', 'llama', '-o', 'jsonpath={.spec.replicas}'])
    suspended = w.command(['kubectl', '-n', 'flux-system', 'get', 'kustomization', 'apps', '-o', 'jsonpath={.spec.suspend}']) == 'true'
    template = w.command(['kubectl', '-n', 'apps', 'get', 'deployment', 'llama', '-o', 'jsonpath={.spec.template}'])
    w.ROOT = ROOT
    w.save('state-before.json', {'replicas': replicas, 'flux_suspended': suspended, 'template_sha256': w.h.digest(template)})
    arm = copy.deepcopy(w.h.load_plan(HERE / 'qualification/plan-full.json')['arms'][0])
    arm['metadata'].update(runtime='vLLM0.31.0 + EXL3 1.5.4 + Orca7ddffb12', image=tm.IMAGE)
    thread = None
    try:
        if not suspended:
            w.command(['flux', 'suspend', 'kustomization', 'apps'])
        w.command(['kubectl', '-n', 'apps', 'scale', 'deployment', 'llama', '--replicas=0'])
        w.command(['kubectl', '-n', 'apps', 'wait', '--for=delete', 'pod', '-l', 'app.kubernetes.io/name=llama', '--timeout=10m'], 660)
        if w.command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']):
            raise RuntimeError('GPU remains occupied')
        profiles = profiles or [('control', False, 64, True, False), ('checkpoint', True, 64, True, True)]
        for name, enabled, unit, run_sessions, run_quality in profiles:
            print('PREFIX START ' + name, flush=True)
            thread = launch(name, selected, enabled, unit, gpu_fraction)
            w.wait_ready()
            warm = {'messages': [{'role': 'user', 'content': 'Reply only READY.'}], 'temperature': 0,
                    'reasoning_effort': 'low', 'max_tokens': 128, 'stream': True, 'stream_options': {'include_usage': True}}
            w.save('warmup.json', w.h.request(arm, warm, 1800))
            logs = subprocess.run(['docker', 'logs', w.active], capture_output=True, text=True, check=True, timeout=90)
            startup = logs.stdout + logs.stderr
            (w.ROOT / 'startup.log').write_text(startup)
            diagnostics = []
            for line in startup.splitlines():
                if 'KV_GROUP_DIAGNOSTIC ' in line:
                    diagnostics.append(json.loads(line.split('KV_GROUP_DIAGNOSTIC ', 1)[1]))
            if not diagnostics:
                raise RuntimeError('Runtime KV geometry diagnostic did not execute')
            info = diagnostics[-1]
            w.save('runtime-kv-groups.json', info)
            expected_unit = unit if unit is not None else info['scheduler_block_size']
            if info['hash_block_size'] != expected_unit or any(g['block_size'] % expected_unit for g in info['groups']):
                raise RuntimeError('Resolved prefix geometry failed validation')
            if enabled and not (info['eagle_group_ids'] and info['enable_partial_hash_hits']
                                and info['num_reprefillable_tokens'] == 0
                                and info['mamba_shared_prefix_checkpoint']
                                and any(m['shared_prefix_checkpoint'] for m in info['managers'])):
                raise RuntimeError('Shared-prefix checkpoint runtime gate not active')
            scheduler_records = [json.loads(line.split('SCHEDULER_CHECKPOINT_DIAGNOSTIC ', 1)[1])
                                 for line in startup.splitlines() if 'SCHEDULER_CHECKPOINT_DIAGNOSTIC ' in line]
            if not scheduler_records:
                raise RuntimeError('Scheduler startup diagnostic did not execute')
            scheduler = scheduler_records[-1]
            w.save('runtime-scheduler.json', scheduler)
            if enabled and not (scheduler['mamba_partial_cache_hit'] and scheduler['mamba_shared_prefix_checkpoint']):
                raise RuntimeError('Scheduler checkpoint gate not active')
            arm['metadata']['launch'] = json.loads((w.ROOT / 'settings.json').read_text())
            if run_sessions:
                session_tests(arm, generation_effort)
            if run_quality:
                q.run(w, arm, w.ROOT / 'orca_vllm-concurrent-quality.jsonl')
                plan = copy.deepcopy(w.h.load_plan(HERE / 'qualification/plan-full.json'))
                plan.pop('sha256')
                plan['arms'] = [copy.deepcopy(arm)]
                plan['arms'][0]['metadata']['launch'] = json.loads((w.ROOT / 'settings.json').read_text())
                plan['jobs'] = [j for j in plan['jobs'] if j['case']['tier'] == 'tool_replay']
                plan['sha256'] = w.h.digest(plan)
                w.save('plan-tools.json', plan)
                w.h.run(plan, arm['name'], w.ROOT / 'tools', 1800)
                w.save('tools-summary.json', w.h.summarize(plan, w.ROOT / 'tools'))
            stop_thread = thread; thread = None
            tm.stop(stop_thread)
    finally:
        failures = []
        try:
            tm.stop(thread)
        except Exception as exc:
            failures.append('cleanup: ' + str(exc))
        for args in ([['kubectl', '-n', 'apps', 'scale', 'deployment', 'llama', '--replicas=' + replicas]] + ([] if suspended else [['flux', 'resume', 'kustomization', 'apps']])):
            try:
                w.command(args)
            except Exception as exc:
                failures.append(str(exc))
        try:
            w.command(['kubectl', '-n', 'apps', 'rollout', 'status', 'deployment/llama', '--timeout=15m'], 960)
            if w.h.digest(w.command(['kubectl', '-n', 'apps', 'get', 'deployment', 'llama', '-o', 'jsonpath={.spec.template}'])) != w.h.digest(template):
                failures.append('Deployment template changed')
        except Exception as exc:
            failures.append(str(exc))
        w.ROOT = ROOT
        w.save('state-restored.json', {'restored': not failures, 'failures': failures})
        print('PREFIX RESTORED ' + str(not failures), flush=True)
        if failures:
            raise RuntimeError('Restoration requires attention')

if __name__ == '__main__':
    main()
