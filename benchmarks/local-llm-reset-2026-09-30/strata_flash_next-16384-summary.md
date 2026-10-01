# Baseline: strata_flash_next

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `7037bfaa64ef760a7c690c59d4f6347849b8a2e208eca122eb860be24babb0cb`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 17/20 | 0.850 | 20 | 109–146 |
| extraction | 4/6 | 0.667 | 5 | 64–84 |
| reasoning | 2/2 | 1.000 | 2 | 73–75 |
| relevance | 1/2 | 0.833 | 2 | 85–87 |
| retrieval | 6/6 | 1.000 | 2 | 1909–114811 |
| tool_replay | 16/16 | 1.000 | 8 | 575–673 |
| transcript_qa | 2/2 | 1.000 | 2 | 83–91 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | 9.935 | 20.242 | 49.600 | 1241.120 |
| decode-short | 1.110 | 11.543 | 49.000 | 357.012 |
| prefill-medium | 8.069 | 8.069 | n/a | 1530.541 |
| prefill-short | 1.117 | 1.117 | n/a | 349.261 |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `code-diff-keys:0:42`: score=0.0; status=ok; finish=length.
- `extract-missing:0:42`: score=0.0; status=ok; finish=stop.
- `extract-quoted:0:42`: score=0.0; status=ok; finish=stop.
- `code-merge-config:0:42`: score=0.0; status=ok; finish=length.
- `support-relevant-db:0:42`: score=0.6666666666666666; status=ok; finish=stop.
- `code-redact:0:42`: score=0.0; status=ok; finish=stop.
