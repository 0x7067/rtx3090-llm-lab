# Baseline: strata_flash_next

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `732c9843f1e2e483f656a0ee61f4ed8371d1d41cca4077d2ec73d4e53a8d37b5`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 12/15 | 0.800 | 5 | 97–116 |
| extraction | 8/9 | 0.889 | 3 | 66–85 |
| reasoning | 3/3 | 1.000 | 1 | 76–79 |
| relevance | 3/3 | 1.000 | 1 | 86–89 |
| tool_replay | 6/6 | 1.000 | 1 | 575–630 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `extract-missing:0:314`: score=0.0; status=ok; finish=stop.
- `code-diff-keys:0:314`: score=0.0; status=ok; finish=stop.
- `code-topological:0:2718`: score=0.0; status=ok; finish=stop.
- `code-diff-keys:0:42`: score=0.0; status=ok; finish=stop.
