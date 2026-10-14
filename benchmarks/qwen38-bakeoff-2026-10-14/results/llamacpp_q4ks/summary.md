# Baseline: llamacpp_q4ks

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `a7a533d2e02c5b2b36df4c979e2468db09be2bb606eaa85a0bf97ff48a236149`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 20/20 | 1.000 | 20 | 109–145 |
| extraction | 6/6 | 1.000 | 5 | 64–86 |
| reasoning | 2/2 | 1.000 | 2 | 71–77 |
| relevance | 2/2 | 1.000 | 2 | 85–88 |
| retrieval | 6/6 | 1.000 | 2 | 1911–114812 |
| tool_replay | 15/16 | 0.938 | 8 | 620–715 |
| transcript_qa | 2/2 | 1.000 | 2 | 82–91 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | 11.665 | 20.991 | 54.798 | 1078.229 |
| decode-short | 0.732 | 9.153 | 60.681 | 715.301 |
| prefill-medium | 13.616 | 13.616 | n/a | 1074.096 |
| prefill-short | 0.733 | 0.733 | n/a | 711.178 |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `episode-unicode-first:0:42`: score=0.0; status=ok; finish=tool_calls.
