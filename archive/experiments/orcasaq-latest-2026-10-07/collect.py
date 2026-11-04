"""Compare matched requests, preserving separate serial and paired quality scopes."""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / 'benchmarks/orca-thinkingcap-2026-10-07'
NEW = HERE / 'qualification'

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def quality(rs):
    qs = [r for r in rs if r['score'] is not None]
    return {'passed': sum(r['score'] == 1 for r in qs), 'count': len(qs),
            'failures': [{'id': r['id'], 'detail': r.get('grade_detail', r.get('detail'))} for r in qs if r['score'] != 1]}

def serial(base):
    rs = rows(base / 'orca_vllm-full/results.jsonl') + rows(base / 'orca_vllm-speed-repeat/results.jsonl')
    groups = {}
    for r in rs:
        if r['score'] is None:
            groups.setdefault(r['id'].split(':')[0], []).append(r['result'])
    return {'quality': quality(rs), 'performance': {
        key: {'n': len(values), 'median_output_tps': statistics.median(r['e2e_output_tps'] for r in values),
              'median_ttft_s': statistics.median(r['ttft_s'] for r in values)}
        for key, values in groups.items()}}

def panel(base):
    bs = rows(base / 'orca_vllm-concurrent-quality.jsonl')
    return {**quality([r for b in bs for r in b['rows']]),
            'inference_seconds': sum(b['inference_wall_s'] for b in bs)}

def concurrency(base, extra=None):
    groups = {}
    bs = rows(base / 'orca_vllm-concurrency.jsonl')
    if extra:
        bs += [b for b in rows(extra / 'orca_vllm-concurrency.jsonl') if not b['retrieval'] and b['callers'] == 2 and b['filler_lines'] == 0]
    for b in bs:
        key = str((b['callers'], b['filler_lines'], b['retrieval'], b.get('warm', False)))
        groups.setdefault(key, []).append(b)
    return {key: {'batches': len(bs), 'median_seconds': statistics.median(b['batch_wall_s'] for b in bs),
                  'median_output_tps': statistics.median(b['aggregate_output_tps'] for b in bs),
                  'retrieval_passes': [r['retrieval_pass'] for b in bs for r in b['rows']],
                  'prompt_tokens': [r['result']['usage']['prompt_tokens'] for b in bs for r in b['rows']]}
            for key, bs in groups.items()}

def requests(base, label):
    return {r['id']: r['request_sha256'] for r in rows(base / f'orca_vllm-{label}/results.jsonl')}

def main():
    audits = {}
    for label in ('full', 'speed-repeat'):
        a, b = requests(OLD, label), requests(NEW, label)
        audits[label] = {'count': len(b), 'identical_ids_and_request_hashes': a == b}
        assert a == b, f'Unmatched {label} requests'
    def panel_hashes(base):
        return {r['id']: r['request_sha256'] for b in rows(base / 'orca_vllm-concurrent-quality.jsonl') for r in b['rows']}
    a, b = panel_hashes(OLD / 'orca-concurrent-quality'), panel_hashes(NEW)
    audits['concurrent_coding'] = {'count': len(b), 'identical_ids_and_request_hashes': a == b}
    assert a == b, 'Unmatched coding panel'
    def long_hashes(base):
        return {(b['repeat'], b.get('warm', False), r['caller']): r['request_sha256']
                for b in rows(base / 'orca_vllm-concurrency.jsonl') if b['retrieval'] for r in b['rows']}
    a, b = long_hashes(OLD / 'orca-concurrent-quality'), long_hashes(NEW)
    audits['long_context'] = {'count': len(b), 'identical_ids_and_request_hashes': a == b}
    assert a == b, 'Unmatched long-context requests'
    result = {'request_audit': audits,
              'scope': 'Latest engines plus publisher preset compared with previous engines plus locally tuned preset; changes are confounded.',
              'previous': {**serial(OLD), 'paired_coding': panel(OLD / 'orca-concurrent-quality'),
                           'concurrency': concurrency(OLD / 'orca-concurrent-quality', OLD)},
              'latest_vendor_preset': {**serial(NEW), 'paired_coding': panel(NEW), 'concurrency': concurrency(NEW)}}
    (HERE / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
