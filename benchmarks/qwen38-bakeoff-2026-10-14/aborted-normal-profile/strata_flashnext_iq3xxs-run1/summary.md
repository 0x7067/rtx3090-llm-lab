# Baseline: strata_flashnext_iq3xxs

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `a7a533d2e02c5b2b36df4c979e2468db09be2bb606eaa85a0bf97ff48a236149`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 0/20 | 0.000 | 20 | None–None |
| extraction | 0/6 | 0.000 | 5 | None–None |
| reasoning | 0/2 | 0.000 | 2 | None–None |
| relevance | 0/2 | 0.000 | 2 | None–None |
| retrieval | 0/6 | 0.000 | 2 | None–None |
| tool_replay | 1/16 | 0.062 | 8 | 629–629 |
| transcript_qa | 1/2 | 0.500 | 2 | 82–82 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | n/a | n/a | n/a | n/a |
| decode-short | n/a | n/a | n/a | n/a |
| prefill-medium | n/a | n/a | n/a | n/a |
| prefill-short | n/a | n/a | n/a | n/a |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `retrieve-revision-4096:0:42`: score=0.0; status=protocol_error; finish=None.
- `code-env-lines:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-reconcile:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `episode-cache-first:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `retrieve-join-4096:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `perf-prefill-medium:0:42`: score=None; status=transport_or_protocol_error; finish=None.
- `episode-untrusted-tool-next:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-diff-keys:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-rolling-window:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `support-reason-boundary:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `episode-pt-pods-next:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `extract-pt-correction:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `episode-pt-pods-first:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `support-relevant-config:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-range-coalesce:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `episode-cache-next:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `retrieve-revision-64:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-retry-budget:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `episode-pod-logs-next:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-pagination:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `extract-missing:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `episode-unicode-first:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `perf-decode-medium:0:42`: score=None; status=transport_or_protocol_error; finish=None.
- `episode-quoted-path-first:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-ttl:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `retrieve-join-64:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `episode-quoted-path-next:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `retrieve-revision-1024:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-sse-frames:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `episode-pod-logs-first:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `extract-boolean:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `episode-not-found-first:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-stable-dedupe:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `episode-failed-test-next:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `extract-quoted:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `episode-untrusted-tool-first:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `retrieve-join-1024:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-histogram:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `support-transcript-pt:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `perf-decode-short:0:42`: score=None; status=transport_or_protocol_error; finish=None.
- `code-merge-config:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `episode-not-found-next:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `extract-en-correction:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-lru:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `episode-failed-test-first:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `support-relevant-db:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-jsonl:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `perf-prefill-short:0:42`: score=None; status=transport_or_protocol_error; finish=None.
- `code-flatten:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-path-normalize:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-batch-bytes:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-rate-limit:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `support-reason-deps:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-topological:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `code-redact:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
- `extract-units:0:42`: score=0.0; status=transport_or_protocol_error; finish=None.
