"""Summarize the bounded sweep and gate the provisional selection."""
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'tuning'
BASE = HERE / 'qualification'

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def conc(base):
    groups = {}
    for b in rows(base / 'orca_vllm-concurrency.jsonl'):
        key = f"lines{b['filler_lines']}-warm{b['warm']}"
        groups.setdefault(key, []).append(b)
    return {key: {'median_seconds': statistics.median(b['batch_wall_s'] for b in bs),
                  'aggregate_output_tps': statistics.median(b['aggregate_output_tps'] for b in bs),
                  'retrieval_checks': [r['retrieval_pass'] for b in bs for r in b['rows'] if r['retrieval_pass'] is not None]}
            for key, bs in groups.items()}

def quality(path):
    rs = [r for r in rows(path) if r['score'] is not None]
    return {'passed': sum(r['score'] == 1 for r in rs), 'count': len(rs),
            'failed_ids': [r['id'] for r in rs if r['score'] != 1]}

def metric(text, name):
    matches = re.findall(r'^' + re.escape(name) + r'(?:\{[^\n]*\})?\s+([\d.eE+-]+)$', text, re.M)
    return sum(map(float, matches)) if matches else None

def screen(name):
    base = ROOT / name
    text = (base / 'metrics-after-screen.txt').read_text()
    accepted = metric(text, 'vllm:spec_decode_num_accepted_tokens_total')
    drafted = metric(text, 'vllm:spec_decode_num_draft_tokens_total')
    pool = re.search(r'kv_cache_size_tokens="(\d+)"', text)
    return {**json.loads((base / 'settings.json').read_text()), **json.loads((base / 'screen.json').read_text()),
            'concurrency': conc(base), 'kv_capacity_tokens': int(pool[1]) if pool else None,
            'spec_acceptance_fraction': accepted / drafted if accepted is not None and drafted else None,
            'preemptions': metric(text, 'vllm:num_preemptions_total')}

def main():
    selected = json.loads((ROOT / 'selection.json').read_text())
    names = [s['name'] for s in json.loads((ROOT / 'screens.json').read_text())]
    screens = {name: screen(name) for name in names}
    # Ensure the screening bodies and seeded requests match every control/candidate.
    def hashes(name):
        base = ROOT / name
        return {'serial': {r['id']: r['request_sha256'] for r in rows(base / 'speed/results.jsonl')},
                'paired': {(b['filler_lines'], b['repeat'], r['caller']): r['request_sha256']
                           for b in rows(base / 'orca_vllm-concurrency.jsonl') for r in b['rows']}}
    control = hashes('control-before')
    audit = {name: hashes(name) == control for name in names}
    assert all(audit.values()), 'Unmatched screening requests'
    full = quality(ROOT / 'selected-validation/full/results.jsonl')
    old = quality(BASE / 'orca_vllm-full/results.jsonl')
    validated = conc(ROOT / 'selected-validation')
    baseline_long = conc(BASE)
    reasons = []
    if full['count'] != 54 or set(full['failed_ids']) - set(old['failed_ids']):
        reasons.append('New serial quality failure or incomplete suite')
    if sum(len(g['retrieval_checks']) for g in validated.values()) != 8 or not all(p for g in validated.values() for p in g['retrieval_checks']):
        reasons.append('Long-context answer check failed or incomplete')
    for key, limit in [('lines5200-warmFalse', 1.05), ('lines5200-warmTrue', 1.10)]:
        if validated[key]['median_seconds'] > baseline_long[key]['median_seconds'] * limit:
            reasons.append(key + ' exceeded validation latency gate')
    proposed = selected['selected']
    recommendation = proposed if not reasons else {'name': 'publisher-control', 'mtp_depth': 2, 'batch_tokens': 2048}
    result = {'screens': screens, 'request_audit': audit, 'selection': selected,
              'selected_full_quality': full, 'baseline_full_quality': old,
              'selected_long_context': validated, 'baseline_long_context': baseline_long,
              'validation_rejections': reasons, 'recommended': recommendation,
              'scope': 'Local screening with two repetitions per context and fixed output; limited fixtures, no claim of general quality equivalence.'}
    (ROOT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    depth = recommendation['mtp_depth']
    extra = '--speculative-config ' + json.dumps({'method': 'qwen3_next_mtp', 'num_speculative_tokens': depth}, separators=(',', ':')) if depth else ''
    (ROOT / 'recommended.env').write_text('PRESET=16gb-mtp\nMAXLEN=150000\nMAXSEQS=4\n'
                                       + f'MAXBATCHTOK={recommendation["batch_tokens"]}\nEXTRA=\'{extra}\'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
