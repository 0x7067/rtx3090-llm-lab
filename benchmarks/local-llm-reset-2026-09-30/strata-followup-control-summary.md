# Baseline: strata_flash_next

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `a5966572be0a43a8e83c85bdeb64035cfa6eb541ce8c06ca2d7c7f1044b3840f`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 6/15 | 0.400 | 5 | 118–139 |
| extraction | 9/9 | 1.000 | 3 | 66–85 |
| reasoning | 3/3 | 1.000 | 1 | 77–78 |
| relevance | 3/3 | 1.000 | 1 | 87–88 |
| tool_replay | 6/6 | 1.000 | 1 | 575–632 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `code-topological:0:314`: score=0.0; status=ok; finish=length.
- `code-redact:0:2718`: score=0.0; status=ok; finish=length.
- `code-merge-config:0:2718`: score=0.0; status=ok; finish=length.
- `code-diff-keys:0:2718`: score=0.0; status=ok; finish=length.
- `code-diff-keys:0:314`: score=0.0; status=ok; finish=length.
- `code-topological:0:2718`: score=0.0; status=ok; finish=length.
- `code-merge-config:0:314`: score=0.0; status=ok; finish=length.
- `code-diff-keys:0:42`: score=0.0; status=ok; finish=length.
- `code-merge-config:0:42`: score=0.0; status=ok; finish=length.
