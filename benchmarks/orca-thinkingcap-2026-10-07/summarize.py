#!/usr/bin/env python3
"""Compact completed campaign evidence; never replace missing observations."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
out = {}
for arm in ('orca_vllm', 'thinkingcap_llama'):
    result = {'quality': {}, 'performance': {}, 'concurrency': {}}
    for label in ('full', 'coding-repeat', 'speed-repeat'):
        path = ROOT / f'{arm}-{label}' / 'results.jsonl'
        plan = json.loads((ROOT / f'plan-{label}.json').read_text())
        if not path.exists():
            result['quality'][label] = {'complete': False, 'observed': 0}
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        quality = [r for r in rows if r['score'] is not None]
        result['quality'][label] = {
            'complete': len(rows) == len(plan['jobs']), 'observed': len(rows), 'expected_observations': len(plan['jobs']),
            'quality_observations': len(quality), 'score': sum(r['score'] for r in quality),
            'wall_s': sum(r['result'].get('wall_s', 0) for r in quality),
            'failures': [{'id': r['id'], 'status': r['result']['status'], 'score': r['score'], 'finish': r['result'].get('finish_reason'), 'detail': r['grade_detail']} for r in quality if r['score'] != 1],
        }
        for row in rows:
            if row['score'] is None:
                result['performance'].setdefault(row['id'].split(':')[0], []).append({k: row['result'].get(k) for k in ('status', 'wall_s', 'ttft_s', 'e2e_output_tps', 'usage')})
    for key, rows in result['performance'].items():
        result['performance'][key] = {'observations': rows, 'median_e2e_tps': statistics.median(r['e2e_output_tps'] for r in rows if r['e2e_output_tps'] is not None) if all(r['status'] == 'ok' for r in rows) else None, 'median_ttft_s': statistics.median(r['ttft_s'] for r in rows if r['ttft_s'] is not None) if all(r['status'] == 'ok' for r in rows) else None}
    path = ROOT / f'{arm}-concurrency.jsonl'
    if path.exists():
        groups = defaultdict(list)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            groups[f"callers={row['callers']},lines={row['filler_lines']},retrieval={row['retrieval']},warm={row.get('warm', False)}"].append(row)
        for key, rows in groups.items():
            requests = [r for batch in rows for r in batch['rows']]
            result['concurrency'][key] = {
                'batches_observed': len(rows), 'requests': len(requests),
                'statuses': [r['result']['status'] for r in requests],
                'median_aggregate_tps': statistics.median(r['aggregate_output_tps'] for r in rows),
                'median_batch_wall_s': statistics.median(r['batch_wall_s'] for r in rows),
                'prompt_tokens': [r['result'].get('usage', {}).get('prompt_tokens') for r in requests],
                'completion_tokens': [r['result'].get('usage', {}).get('completion_tokens') for r in requests],
                'retrieval_passes': [r['retrieval_pass'] for r in requests if r['retrieval_pass'] is not None],
            }
    out[arm] = result
(ROOT / 'observations.json').write_text(json.dumps(out, indent=2) + '\n')
for arm, result in out.items():
    print(arm)
    for label, values in result['quality'].items():
        print(label, values.get('score'), '/', values.get('quality_observations'), 'complete=', values['complete'])
        for failure in values.get('failures', []):
            print('failure', failure['id'], failure['status'], failure['score'])
    for key, values in result['performance'].items():
        print(key, 'median E2E', values['median_e2e_tps'], 'TTFT', values['median_ttft_s'])
    for key, values in result['concurrency'].items():
        print(key, 'aggregate', values['median_aggregate_tps'], 'retrieval', values['retrieval_passes'])
