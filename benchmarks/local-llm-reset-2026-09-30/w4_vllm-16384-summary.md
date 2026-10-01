# Baseline: w4_vllm

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `7037bfaa64ef760a7c690c59d4f6347849b8a2e208eca122eb860be24babb0cb`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 20/20 | 1.000 | 20 | 109–146 |
| extraction | 6/6 | 1.000 | 5 | 64–84 |
| reasoning | 2/2 | 1.000 | 2 | 73–75 |
| relevance | 2/2 | 1.000 | 2 | 85–87 |
| retrieval | 4/6 | 0.667 | 2 | 1909–28791 |
| tool_replay | 16/16 | 1.000 | 8 | 620–718 |
| transcript_qa | 2/2 | 1.000 | 2 | 83–91 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | 12.779 | 24.727 | n/a | n/a |
| decode-short | 0.428 | 12.124 | n/a | n/a |
| prefill-medium | 12.776 | 12.776 | n/a | n/a |
| prefill-short | 0.400 | 0.400 | n/a | n/a |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `retrieve-revision-4096:0:42`: score=0.0; status=http_error; finish=None.
- `retrieve-join-4096:0:42`: score=0.0; status=http_error; finish=None.
