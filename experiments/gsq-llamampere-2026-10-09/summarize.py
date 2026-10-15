#!/usr/bin/env python3
"""Summarize completed raw evidence; never rerun inference."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

screen = {}
for path in sorted((ROOT/'screen').glob('*-concurrency.jsonl')):
    name = path.name.removesuffix('-concurrency.jsonl')
    records = rows(path)
    metrics = {}
    for callers, lines in ((1, 0), (2, 0), (2, 2400)):
        selected = [x for x in records if x['callers'] == callers and x['filler_lines'] == lines]
        metrics[f'{callers}_callers_{lines}_lines'] = {
            'aggregate_tps_mean': statistics.mean(x['aggregate_output_tps'] for x in selected),
            'wall_s': [x['batch_wall_s'] for x in selected],
            'prompt_tokens': [[r['result']['usage']['prompt_tokens'] for r in x['rows']] for x in selected],
            'all_requests_ok': all(r['result']['status'] == 'ok' for x in selected for r in x['rows']),
        }
    telemetry = rows(next((ROOT/'screen').glob(f'gsq-resident-{name}-*-telemetry.jsonl')))
    metrics['gpu_peak_mib'] = max(int(x['gpu'].split(',')[0]) for x in telemetry if x['gpu'])
    metrics['paused_samples'] = sum(x.get('paused', False) for x in telemetry)
    screen[name] = metrics

qualification = ROOT/'qualification-ampere-mtp2-f32'
coding = [r for x in rows(qualification/'ampere-mtp2-f32-concurrent-quality.jsonl') for r in x['rows']]
tools = rows(qualification/'ampere-mtp2-f32-tools/results.jsonl')
long = rows(qualification/'ampere-mtp2-f32-long-2x9300.jsonl')
retrieval = [r for x in long for r in x['rows'] if r['pass'] is not None]
telemetry = [r for path in qualification.glob('*-telemetry.jsonl') for r in rows(path)]
historical_path = ROOT.parents[1]/'benchmarks/gsq-resident-2026-10-08/gsq_q8_resident-long-2x9300.jsonl'
historical = rows(historical_path)
result = {
    'screen': screen,
    'qualification': {
        'coding': {'passed': sum(r['score'] == 1 for r in coding), 'total': len(coding)},
        'tools': {'passed': sum(r['score'] == 1 for r in tools), 'total': len(tools)},
        'long_retrieval': {'passed': sum(r['pass'] for r in retrieval), 'total': len(retrieval)},
        'cold_wall_s': long[0]['batch_wall_s'],
        'cold_prompt_tokens': [r['result']['usage']['prompt_tokens'] for r in long[0]['rows']],
        'followup_wall_mean_s': statistics.mean(r['batch_wall_s'] for r in long if r['label'].startswith('followup')),
        'cached_generation': {k: long[-1][k] for k in ('batch_wall_s', 'aggregate_output_tps')},
        'cached_prompt_tokens': [r['result']['usage']['prompt_tokens'] for r in long[-1]['rows']],
        'gpu_peak_mib': max(int(x['gpu'].split(',')[0]) for x in telemetry if x['gpu']),
        'paused_samples': sum(x.get('paused', False) for x in telemetry),
        'thermal_trips': sum(x.get('thermal_trip', False) for x in telemetry),
        'swap_peak_bytes': max(int(x.get('cgroup', {}).get('memory.swap.current', '0')) for x in telemetry),
    },
    'historical_non_speculative_long_reference': {
        'source': str(historical_path),
        'caveat': 'Previous campaign, microbatch512 and non-MTP file; not a fresh matched timing control.',
        'cold_wall_s': historical[0]['batch_wall_s'],
        'followup_wall_mean_s': statistics.mean(x['batch_wall_s'] for x in historical if x['label'].startswith('followup')),
        'cached_generation': {k: historical[-1][k] for k in ('batch_wall_s', 'aggregate_output_tps')},
    },
    'restoration': json.loads((qualification/'restoration.json').read_text()),
}
(ROOT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
