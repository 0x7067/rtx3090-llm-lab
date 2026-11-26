# Baseline: ctl_llamacpp_gsq

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
| decode-medium | 10.823 | 19.911 | 56.228 | 1171.794 |
| decode-short | 0.774 | 9.226 | 60.457 | 712.716 |
| prefill-medium | 13.590 | 13.590 | n/a | 1160.370 |
| prefill-short | 0.782 | 0.782 | n/a | 698.555 |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

None.
