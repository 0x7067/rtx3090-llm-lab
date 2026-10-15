#!/usr/bin/env python3
"""Freeze a matched qualification campaign without altering historical results."""
import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'local-llm-reset-2026-09-30'
HARNESS = ROOT.parent / 'paired-harness/harness.py'
spec = importlib.util.spec_from_file_location('paired_harness', HARNESS)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

arms = []
for original in json.loads((OLD / 'arms-post-upgrade.json').read_text()):
    if original['name'] not in {'orca_vllm', 'thinkingcap_llama'}:
        continue
    a = copy.deepcopy(original)
    a['base_url'] = 'http://127.0.0.1:18185/v1'
    a['metadata']['gpu_profile'] = 'Existing host profile, unchanged; record actual clocks and guard state'
    if a['name'] == 'orca_vllm':
        a['metadata']['launch'] = 'MTP 3, batch tokens 4096, max sequences 8, max context 150000, FP8 KV, prefix cache; pinned retained vLLM image'
    else:
        a['metadata']['launch'] = 'b11277, Q8 K/V, unified 150000-token pool, parallel 2, no speculation, flash attention, full GPU offload'
    arms.append(a)

suite = copy.deepcopy(json.loads((OLD / 'plan-16384.json').read_text())['suite'])
suite['description'] = 'OrcaSAQ2 production speed settings versus ThinkingCap two-slot Q8 unified cache. Authored development regression suite; not held-out production data.'
for c in suite['cases']:
    if c['id'] == 'code-redact':
        c['request']['messages'][0]['content'] += ' The sensitive argument is a list of strings.'
    if c['id'] == 'code-sse-frames':
        c['request']['messages'][0]['content'] += ' Skip events with no data fields; a data field with an empty value still produces an empty payload.'
    if c['id'] == 'extract-quoted':
        c['request']['messages'][0]['content'] += ' Use the literal JSON key "value".'

for label, seeds, cases in [
    ('full', [42], suite['cases']),
    ('coding-repeat', [314, 2718], [c for c in suite['cases'] if c['tier'] == 'coding']),
    ('speed-repeat', [314, 2718], [c for c in suite['cases'] if c['phase'] == 'performance']),
]:
    selected = {**suite, 'seeds': seeds, 'cases': cases}
    path = ROOT / f'plan-{label}.json'
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite {path}')
    h.write_new(path, h.make_plan(selected, arms, f'orca-thinkingcap-{label}-2026-10-07'))
print('Frozen full suite, two extra coding seeds and two extra performance seeds.')
