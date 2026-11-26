# Baseline: nariq_3p0

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `3f158e259ce5de67231a48175452ac8874e13d0ec2f7fede751734605218b0e1`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 20/20 | 1.000 | 20 | 109–145 |
| extraction | 6/6 | 1.000 | 5 | 64–86 |
| reasoning | 2/2 | 1.000 | 2 | 70–77 |
| relevance | 2/2 | 1.000 | 2 | 87–91 |
| retrieval | 6/6 | 1.000 | 2 | 1910–114809 |
| tool_replay | 16/16 | 1.000 | 8 | 622–716 |
| transcript_qa | 2/2 | 1.000 | 2 | 81–90 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | 10.162 | 23.230 | 39.105 | 1218.914 |
| decode-short | 0.627 | 12.653 | 42.490 | 773.743 |
| prefill-medium | 11.797 | 11.797 | n/a | 1217.948 |
| prefill-short | 0.559 | 0.560 | n/a | 780.111 |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

None.
