# Baseline: strata_flashnext_iq3xxs

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `a7a533d2e02c5b2b36df4c979e2468db09be2bb606eaa85a0bf97ff48a236149`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 17/20 | 0.850 | 20 | 109–145 |
| extraction | 5/6 | 0.833 | 5 | 64–86 |
| reasoning | 2/2 | 1.000 | 2 | 71–77 |
| relevance | 2/2 | 1.000 | 2 | 85–88 |
| retrieval | 6/6 | 1.000 | 2 | 1911–114812 |
| tool_replay | 16/16 | 1.000 | 8 | 575–670 |
| transcript_qa | 2/2 | 1.000 | 2 | 82–91 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | 7.837 | 16.093 | 61.900 | 1567.280 |
| decode-short | 1.071 | 9.818 | 58.400 | 366.920 |
| prefill-medium | 7.895 | 7.896 | n/a | 1556.527 |
| prefill-short | 1.089 | 1.101 | n/a | 358.376 |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `code-env-lines:0:42`: score=0.0; status=ok; finish=stop.
- `code-diff-keys:0:42`: score=0.0; status=ok; finish=length.
- `extract-en-correction:0:42`: score=0.0; status=ok; finish=stop.
- `code-redact:0:42`: score=0.0; status=ok; finish=stop.
