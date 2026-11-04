"""Audit checkpoint observations and produce a reviewable serving preset."""
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'prefix-checkpoint'

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def observation(name, root=ROOT):
    base = root / name
    sessions = rows(base / 'sessions.jsonl')
    rs = [r for b in sessions for r in b['rows']]
    warm = [r for b in sessions if b['turn'] for r in b['rows']]
    gen = json.loads((base / 'long-generation.json').read_text())
    details = [r['result']['usage'].get('prompt_tokens_details', {}) for r in warm]
    cached = [d['cached_tokens'] for d in details if d.get('cached_tokens') is not None]
    times = sorted(b['batch_wall_s'] for b in sessions if b['turn'])
    text = (base / 'metrics-after-sessions.txt').read_text()
    pool = re.search(r'kv_cache_size_tokens="(\d+)"', text)
    return {'settings': json.loads((base / 'settings.json').read_text()),
            'runtime_kv_groups': json.loads((base / 'runtime-kv-groups.json').read_text()),
            'runtime_scheduler': json.loads((base / 'runtime-scheduler.json').read_text()),
            'reported_kv_capacity_tokens': int(pool[1]) if pool else None,
            'sessions': {'turns_per_agent': len(sessions), 'passed': sum(r['passed'] for r in rs), 'count': len(rs),
                         'cold_pair_seconds': sessions[0]['batch_wall_s'],
                         'followup_pair_median_seconds': statistics.median(b['batch_wall_s'] for b in sessions if b['turn']),
                         'followup_pair_p95_seconds': times[math.ceil(.95 * len(times)) - 1],
                         'followup_pair_max_seconds': max(times),
                         'followup_agent_ttft_median_seconds': statistics.median(r['result']['ttft_s'] for r in warm),
                         'followup_total_seconds': sum(b['batch_wall_s'] for b in sessions if b['turn']),
                         'followup_cached_tokens_median': statistics.median(cached) if cached else None,
                         'followup_cached_tokens_observed': len(cached),
                         'prompt_tokens_range': [min(r['result']['usage']['prompt_tokens'] for r in rs), max(r['result']['usage']['prompt_tokens'] for r in rs)]},
            'long_generation': {'pair_seconds': gen['batch_wall_s'], 'aggregate_output_tps': gen['aggregate_output_tps'],
                                'scope': ('Stable low effort with cached long histories; wall includes residual prefill and generation.'
                                          if gen.get('generation_effort') == 'low' else
                                          'Low-to-medium effort transition: includes fresh prefill; not isolated warm decoding.'),
                                'cached_tokens': [r['result']['usage'].get('prompt_tokens_details', {}).get('cached_tokens') for r in gen['rows']],
                                'ttft_seconds': [r['result']['ttft_s'] for r in gen['rows']],
                                'completion_tokens': [r['result']['usage']['completion_tokens'] for r in gen['rows']],
                                'finish_reasons': [r['result']['finish_reason'] for r in gen['rows']]}}

def quality(base):
    panel = [r for b in rows(base / 'orca_vllm-concurrent-quality.jsonl') for r in b['rows']]
    tools = rows(base / 'tools/results.jsonl')
    return {'paired_coding_passed': sum(r['score'] == 1 for r in panel), 'paired_coding_count': len(panel),
            'tool_replay_passed': sum(r['score'] == 1 for r in tools), 'tool_replay_count': len(tools),
            'coding_failed_ids': [r['id'] for r in panel if r['score'] != 1],
            'tool_failed_ids': [r['id'] for r in tools if r['score'] != 1]}

def gate(control, candidate, qual):
    reasons = []
    for data in (control, candidate):
        if data['sessions']['passed'] != data['sessions']['count'] or data['sessions']['count'] != 42:
            reasons.append('Growing-session checks failed or incomplete')
    if qual['paired_coding_passed'] != qual['paired_coding_count'] or qual['paired_coding_count'] != 24:
        reasons.append('Paired coding gate failed')
    if qual['tool_replay_passed'] != qual['tool_replay_count'] or qual['tool_replay_count'] != 16:
        reasons.append('Tool replay gate failed')
    if candidate['sessions']['followup_pair_median_seconds'] > control['sessions']['followup_pair_median_seconds'] * .95:
        reasons.append('Cached latency gain below five percent')
    if any(data['sessions']['followup_cached_tokens_observed'] != 40 for data in (control, candidate)):
        reasons.append('Incomplete per-request cached-token evidence')
    elif candidate['sessions']['followup_cached_tokens_median'] <= control['sessions']['followup_cached_tokens_median']:
        reasons.append('No demonstrated incremental cached-token reuse')
    if candidate['sessions']['cold_pair_seconds'] > control['sessions']['cold_pair_seconds'] * 1.05:
        reasons.append('Cold pair exceeded five percent latency regression gate')
    if candidate['sessions']['followup_pair_p95_seconds'] > control['sessions']['followup_pair_p95_seconds'] * 1.10:
        reasons.append('Follow-up tail latency exceeded ten percent regression gate')
    if any(x != 512 for data in (control, candidate) for x in data['long_generation']['completion_tokens']):
        reasons.append('Fixed-budget generation probe ended early')
    if candidate['long_generation']['pair_seconds'] > control['long_generation']['pair_seconds'] * 1.05:
        reasons.append('Long generation exceeded five percent latency regression gate')
    return reasons

def main():
    default = observation('default', HERE / 'prefix-default')
    fine, checkpoint = observation('control'), observation('checkpoint')
    locations = {'default': HERE / 'prefix-default/default', 'fine_hash': ROOT / 'control', 'checkpoint': ROOT / 'checkpoint'}
    def hashes(base):
        return {(b['turn'], r['caller']): r['request_sha256'] for b in rows(base / 'sessions.jsonl') for r in b['rows']}
    reference = hashes(locations['default'])
    identical = {name: hashes(base) == reference for name, base in locations.items()}
    assert all(identical.values()), 'Growing-session request mismatch'
    def gen_hashes(base):
        return [r['request_sha256'] for r in json.loads((base / 'long-generation.json').read_text())['rows']]
    assert all(gen_hashes(base) == gen_hashes(locations['default']) for base in locations.values()), 'Generation request mismatch'
    qs = {'fine_hash': quality(HERE / 'prefix-default/fine-verify'), 'checkpoint': quality(ROOT / 'checkpoint')}
    rejected = {'fine_hash': gate(default, fine, qs['fine_hash']), 'checkpoint': gate(default, checkpoint, qs['checkpoint'])}
    eligible = [name for name, failures in rejected.items() if not failures]
    data = {'default': default, 'fine_hash': fine, 'checkpoint': checkpoint}
    chosen = min(eligible, key=lambda name: data[name]['sessions']['followup_pair_median_seconds']) if eligible else 'default'
    # Prefer the simpler fine-hash setting unless checkpointing demonstrates a
    # further five percent improvement over its identical-geometry control.
    if chosen == 'checkpoint' and 'fine_hash' in eligible and checkpoint['sessions']['followup_pair_median_seconds'] > fine['sessions']['followup_pair_median_seconds'] * .95:
        chosen = 'fine_hash'
    result = {'observations': data, 'quality': qs, 'identical_growing_request_hashes': identical,
              'identical_generation_request_hashes': True, 'rejection_reasons': rejected,
              'recommended_profile': chosen,
              'scope': 'Two independent ~83K growing sessions, twenty follow-up pairs, one cold retrieval and one cold effort-transition512-output-token pair per arm; no isolated warm long decoding, forced eviction or mixed prefill/decode test.'}
    (ROOT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    settings = json.loads((HERE / 'tuning/results.json').read_text())['recommended']
    flags = '--speculative-config ' + json.dumps({'method': 'qwen3_next_mtp', 'num_speculative_tokens': settings['mtp_depth']}, separators=(',', ':'))
    if chosen != 'default':
        flags += ' --mamba-cache-mode align --prefix-match-unit 64'
    if chosen == 'checkpoint':
        flags += ' --enable-mamba-shared-prefix-checkpoint'
    (ROOT / 'recommended.env').write_text('PRESET=16gb-mtp\nMAXLEN=150000\nMAXSEQS=4\n'
                                       + f'MAXBATCHTOK={settings["batch_tokens"]}\nEXTRA=\'{flags}\'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
