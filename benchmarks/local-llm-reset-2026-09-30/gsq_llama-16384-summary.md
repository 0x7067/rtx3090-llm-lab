# Baseline: gsq_llama

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `7037bfaa64ef760a7c690c59d4f6347849b8a2e208eca122eb860be24babb0cb`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 19/20 | 0.950 | 20 | 109–146 |
| extraction | 6/6 | 1.000 | 5 | 64–84 |
| reasoning | 2/2 | 1.000 | 2 | 73–75 |
| relevance | 2/2 | 1.000 | 2 | 85–87 |
| retrieval | 6/6 | 1.000 | 2 | 1909–114811 |
| tool_replay | 16/16 | 1.000 | 8 | 620–718 |
| transcript_qa | 2/2 | 1.000 | 2 | 83–91 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | 12.008 | 25.920 | 36.732 | 1046.650 |
| decode-short | 0.659 | 13.531 | 39.697 | 760.320 |
| prefill-medium | 13.428 | 13.429 | n/a | 1047.428 |
| prefill-short | 0.699 | 0.699 | n/a | 750.358 |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `code-sse-frames:0:42`: score=0.0; status=ok; finish=stop.
