# Baseline: thinkingcap_llama

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `7037bfaa64ef760a7c690c59d4f6347849b8a2e208eca122eb860be24babb0cb`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 20/20 | 1.000 | 20 | 109–146 |
| extraction | 6/6 | 1.000 | 5 | 64–84 |
| reasoning | 2/2 | 1.000 | 2 | 73–75 |
| relevance | 2/2 | 1.000 | 2 | 85–87 |
| retrieval | 6/6 | 1.000 | 2 | 1909–114811 |
| tool_replay | 16/16 | 1.000 | 8 | 620–718 |
| transcript_qa | 2/2 | 1.000 | 2 | 83–91 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | 11.972 | 25.089 | 38.955 | 1049.703 |
| decode-short | 0.649 | 12.747 | 42.239 | 779.650 |
| prefill-medium | 13.424 | 13.424 | n/a | 1050.155 |
| prefill-short | 0.651 | 0.651 | n/a | 774.785 |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

None.
