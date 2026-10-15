"""Compare the memory-safe baseline and fine-hash candidate, retaining failed runs."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'memory-qualified'
spec = importlib.util.spec_from_file_location('prefix_observations', HERE / 'collect-prefix.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

def main():
    default = p.observation('default-safe', ROOT)
    fine = p.observation('fine-safe', ROOT)
    def hashes(base):
        return {(b['turn'], r['caller']): r['request_sha256']
                for b in p.rows(base / 'sessions.jsonl') for r in b['rows']}
    assert hashes(ROOT / 'default-safe') == hashes(ROOT / 'fine-safe'), 'Growing request mismatch'
    def gen(base):
        return json.loads((base / 'long-generation.json').read_text())
    a, b = gen(ROOT / 'default-safe'), gen(ROOT / 'fine-safe')
    assert [r['request_sha256'] for r in a['rows']] == [r['request_sha256'] for r in b['rows']], 'Generation request mismatch'
    assert all(r['result']['usage']['prompt_tokens_details']['cached_tokens'] > 0 for batch in (a, b) for r in batch['rows']), 'Generation was not cached'
    qual = p.quality(ROOT / 'fine-safe')
    reasons = p.gate(default, fine, qual)
    selected = fine if not reasons else default
    speedup = default['sessions']['followup_pair_median_seconds'] / fine['sessions']['followup_pair_median_seconds']
    result = {'default_safe': default, 'fine_safe': fine, 'fine_quality': qual,
              'identical_growing_and_generation_request_hashes': True,
              'cache_followup_speedup': speedup, 'fine_rejection_reasons': reasons,
              'recommended': selected['settings'],
              'prior_095_failure': {'root': '../prefix-default/default', 'completed_pairs': 5,
                                    'symptom': 'CUDA OOM in Orca MTP shard_gemm reconstruction, 170MiB allocation; engine died on next growing pair.',
                                    'scope': 'Infrastructure failure; not a model quality score. Individual failed response bodies were not saved in this first run; server stack and scheduler dump were saved.'},
              'scope': 'Two independent83K histories, twenty growing follow-up pairs, cached512-output-token generation pair, then24 coding and16 tool checks; no forced eviction, longer two-request histories, or isolated decode-only timing.'}
    (ROOT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    settings = selected['settings']
    flags = '--speculative-config ' + json.dumps({'method': 'qwen3_next_mtp', 'num_speculative_tokens': settings['mtp_depth']}, separators=(',', ':'))
    if not reasons:
        flags += ' --mamba-cache-mode align --prefix-match-unit 64'
    (ROOT / 'recommended.env').write_text('PRESET=16gb-mtp\nGPU_FRAC=0.92\nMAXLEN=150000\nMAXSEQS=4\n'
                                       + f'MAXBATCHTOK={settings["batch_tokens"]}\nEXTRA=\'{flags}\'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
