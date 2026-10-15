#!/usr/bin/env python3
"""Qualify ThinkingCap's larger Q8 pool against the preserved Orca observations."""
import importlib.util
import hashlib
import json
from pathlib import Path
import statistics
import threading

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
spec = importlib.util.spec_from_file_location('window', SOURCE / 'run.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
w.ROOT = ROOT
qs = importlib.util.spec_from_file_location('concurrent_quality', SOURCE / 'concurrent_quality.py')
q = importlib.util.module_from_spec(qs)
qs.loader.exec_module(q)
original_concurrency = w.concurrency


def concurrency(arm):
    original_concurrency(arm)
    q.run(w, arm, ROOT / 'thinkingcap_llama-concurrent-quality.jsonl')


def prepare():
    for label in ('full', 'coding-repeat', 'speed-repeat'):
        plan = w.h.load_plan(SOURCE / f'plan-{label}.json')
        plan.pop('sha256')
        plan['arms'] = [a for a in plan['arms'] if a['name'] == 'thinkingcap_llama']
        plan['arms'][0]['metadata']['launch'] = 'b11277, Q8 K/V, unified 200000-token pool, parallel 2, no speculation, flash attention, full GPU offload'
        # Preserve every request and grader byte. Only arm provenance changes.
        plan['sha256'] = w.h.digest(plan)
        w.save(f'plan-{label}.json', plan)


def launch(arm):
    w.active = 'orca-thinkingcap-thinkingcap-200k'
    args = ['docker', 'run', '-d', '--name', w.active, '--gpus', 'all', '--network', 'host', '--memory', '32g', '--memory-swap', '32g', '--cpus', '8',
            '-v', '/data/models/qwen3.8/mradermacher/ThinkingCap-Qwen3.8-27B-abliterated-GGUF:/models:ro', w.LLAMA,
            '--model', '/models/ThinkingCap-Qwen3.8-27B-abliterated.Q4_K_S.gguf', '--ctx-size', '200000', '--parallel', '2', '--kv-unified',
            '--n-gpu-layers', '999', '--flash-attn', 'on', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0', '--jinja', '--host', '127.0.0.1', '--port', str(w.PORT), '--alias', arm['model'], '--threads', '8', '--threads-batch', '8']
    w.save(f"{arm['name']}-launch.json", args)
    w.command(args)
    w.stop_monitor.clear(); w.abort.clear()
    thread = threading.Thread(target=w.monitor, args=(w.active, ROOT / f"{arm['name']}-telemetry.jsonl"))
    thread.start()
    return thread


def compare_cross_plan(plan, control_path, candidate_path):
    label = candidate_path.name.removeprefix('orca_vllm-')
    source_plan = w.h.load_plan(SOURCE / f'plan-{label}.json')
    assert plan['jobs'] == source_plan['jobs'], 'Request or grader mismatch'
    assert plan['suite'] == source_plan['suite'], 'Scoring or analysis settings changed'
    control = w.h.load_run(control_path, plan)
    candidate = w.h.load_run(SOURCE / candidate_path.name, source_plan)
    pairs = []
    performance = {}
    for job in plan['jobs']:
        a, b = control[job['id']], candidate[job['id']]
        assert a['request_sha256'] == b['request_sha256'] == job['request_sha256']
        if job['case']['phase'] == 'quality':
            pairs.append({'id': job['id'], 'tier': job['case']['tier'], 'thinkingcap_score': a['score'], 'orca_score': b['score'], 'thinkingcap_wall_s': a['result'].get('wall_s'), 'orca_wall_s': b['result'].get('wall_s')})
        else:
            performance.setdefault(job['case']['tier'], []).append({'id': job['id'], 'thinkingcap': {key: a['result'].get(key) for key in ('status', 'wall_s', 'ttft_s', 'e2e_output_tps', 'usage')}, 'orca': {key: b['result'].get(key) for key in ('status', 'wall_s', 'ttft_s', 'e2e_output_tps', 'usage')}})
    return {'thinkingcap_plan_sha256': plan['sha256'], 'orca_plan_sha256': source_plan['sha256'], 'matched_request_and_grader_bytes': True,
            'quality_observations': len(pairs), 'thinkingcap_score': sum(p['thinkingcap_score'] for p in pairs), 'orca_score': sum(p['orca_score'] for p in pairs),
            'orca_only_failures': [p for p in pairs if p['orca_score'] < p['thinkingcap_score']], 'thinkingcap_only_failures': [p for p in pairs if p['thinkingcap_score'] < p['orca_score']],
            'pairs': pairs, 'performance': performance, 'interpretation': 'Descriptive paired observations across separately frozen arm metadata. No statistical equivalence claim.'}


if __name__ == '__main__':
    prepare()
    model = Path('/data/models/qwen3.8/mradermacher/ThinkingCap-Qwen3.8-27B-abliterated-GGUF/ThinkingCap-Qwen3.8-27B-abliterated.Q4_K_S.gguf')
    with model.open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    expected = '1cd114864f9ab26a508e3943b55b0c38b556ac368f618dbac63602a00e02d0ed'
    if actual != expected:
        raise RuntimeError('ThinkingCap model checksum mismatch')
    w.save('model-checksum.json', {'sha256': actual, 'expected_sha256': expected, 'size': model.stat().st_size})
    w.launch = launch
    w.concurrency = concurrency
    w.h.compare = compare_cross_plan
    w.main()
