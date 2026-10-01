# Baseline: orca_vllm

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `186ca9720d0c90d16f0d9dd1ca10505bd81c61b4da72f4c5f95ccc3990fe081d`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 19/20 | 0.950 | 20 | 108–144 |
| extraction | 6/6 | 1.000 | 5 | 64–83 |
| reasoning | 2/2 | 1.000 | 2 | 71–79 |
| relevance | 2/2 | 1.000 | 2 | 89–90 |
| retrieval | 6/6 | 1.000 | 2 | 1910–114809 |
| tool_replay | 16/16 | 1.000 | 8 | 622–715 |
| transcript_qa | 2/2 | 1.000 | 2 | 81–88 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | 11.371 | 19.627 | n/a | n/a |
| decode-short | 0.494 | 8.815 | n/a | n/a |
| prefill-medium | 11.315 | 11.315 | n/a | n/a |
| prefill-short | 0.451 | 0.451 | n/a | n/a |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `code-rate-limit:0:42`: score=0.0; status=ok; finish=stop.
