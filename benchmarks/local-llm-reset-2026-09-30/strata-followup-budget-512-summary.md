# Baseline: strata_flash_next

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `497864bfcfc4288f037da57680455e2fc5865a65ccf835b1362a5cbd963471b4`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 6/15 | 0.400 | 5 | 121–140 |
| extraction | 9/9 | 1.000 | 3 | 65–85 |
| reasoning | 3/3 | 1.000 | 1 | 77–79 |
| relevance | 3/3 | 1.000 | 1 | 87–90 |
| tool_replay | 6/6 | 1.000 | 1 | 572–631 |

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
- `code-topological:0:2718`: score=0.0; status=ok; finish=stop.
- `code-topological:0:42`: score=0.0; status=ok; finish=length.
- `code-merge-config:0:314`: score=0.0; status=ok; finish=length.
- `code-diff-keys:0:42`: score=0.0; status=ok; finish=length.
