# Baseline: strata_flash_next

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `1e599ff31f7c23701d1bbc98cb4b7ec6092e91fafad6be6c69196a4b081cad56`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 6/15 | 0.400 | 5 | 120–138 |
| extraction | 9/9 | 1.000 | 3 | 67–86 |
| reasoning | 3/3 | 1.000 | 1 | 78–79 |
| relevance | 2/3 | 0.667 | 1 | 88–91 |
| tool_replay | 6/6 | 1.000 | 1 | 576–632 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `code-topological:0:314`: score=0.0; status=ok; finish=length.
- `code-merge-config:0:2718`: score=0.0; status=ok; finish=length.
- `support-relevant-db:0:314`: score=0.0; status=ok; finish=stop.
- `code-diff-keys:0:2718`: score=0.0; status=ok; finish=length.
- `code-diff-keys:0:314`: score=0.0; status=ok; finish=length.
- `code-topological:0:2718`: score=0.0; status=ok; finish=length.
- `code-topological:0:42`: score=0.0; status=ok; finish=length.
- `code-merge-config:0:314`: score=0.0; status=ok; finish=length.
- `code-diff-keys:0:42`: score=0.0; status=ok; finish=length.
- `code-merge-config:0:42`: score=0.0; status=ok; finish=length.
