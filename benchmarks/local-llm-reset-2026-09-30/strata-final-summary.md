# Baseline: strata_flash_next

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `957a049310aa926e85af6a96f32f6145ba67691530b2983ae208c89cfb0872bc`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 11/20 | 0.550 | 20 | 109–145 |
| extraction | 5/6 | 0.833 | 5 | 64–84 |
| reasoning | 2/2 | 1.000 | 2 | 73–78 |
| relevance | 2/2 | 1.000 | 2 | 89–91 |
| retrieval | 5/6 | 0.833 | 2 | 1910–114812 |
| tool_replay | 16/16 | 1.000 | 8 | 575–670 |
| transcript_qa | 2/2 | 1.000 | 2 | 83–89 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | 7.542 | 18.284 | 47.600 | 1638.287 |
| decode-short | 1.105 | 10.344 | 55.300 | 355.035 |
| prefill-medium | 7.635 | 7.642 | n/a | 1617.929 |
| prefill-short | 1.071 | 1.071 | n/a | 367.479 |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `retrieve-join-4096:0:42`: score=0.0; status=ok; finish=stop.
- `code-diff-keys:0:42`: score=0.0; status=ok; finish=length.
- `code-range-coalesce:0:42`: score=0.0; status=ok; finish=length.
- `extract-missing:0:42`: score=0.0; status=ok; finish=stop.
- `code-sse-frames:0:42`: score=0.0; status=ok; finish=length.
- `code-histogram:0:42`: score=0.0; status=ok; finish=length.
- `code-merge-config:0:42`: score=0.0; status=ok; finish=length.
- `code-flatten:0:42`: score=0.0; status=ok; finish=length.
- `code-rate-limit:0:42`: score=0.0; status=ok; finish=length.
- `code-topological:0:42`: score=0.0; status=ok; finish=length.
- `code-redact:0:42`: score=0.0; status=ok; finish=length.
