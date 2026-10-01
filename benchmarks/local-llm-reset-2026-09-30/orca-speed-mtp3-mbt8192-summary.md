# Baseline: orca_vllm

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `309ac0d2620b6c32beaa558305ec4dac0027d0e49d3d01ef4ac146e93cc23be0`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | n/a | n/a | n/a | n/a |
| decode-short | n/a | n/a | n/a | n/a |
| prefill-medium | n/a | n/a | n/a | n/a |
| prefill-short | n/a | n/a | n/a | n/a |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `perf-decode-medium:0:314`: score=None; status=protocol_error; finish=None.
- `perf-decode-short:0:314`: score=None; status=http_error; finish=None.
- `perf-prefill-medium:0:42`: score=None; status=http_error; finish=None.
- `perf-prefill-short:0:2718`: score=None; status=http_error; finish=None.
- `perf-decode-short:0:2718`: score=None; status=http_error; finish=None.
- `perf-prefill-medium:0:314`: score=None; status=http_error; finish=None.
- `perf-decode-medium:0:2718`: score=None; status=http_error; finish=None.
- `perf-decode-medium:0:42`: score=None; status=http_error; finish=None.
- `perf-prefill-short:0:314`: score=None; status=http_error; finish=None.
- `perf-prefill-short:0:42`: score=None; status=http_error; finish=None.
- `perf-decode-short:0:42`: score=None; status=http_error; finish=None.
- `perf-prefill-medium:0:2718`: score=None; status=http_error; finish=None.
