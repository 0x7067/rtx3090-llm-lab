# Baseline: thinkingcap_llama

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `186ca9720d0c90d16f0d9dd1ca10505bd81c61b4da72f4c5f95ccc3990fe081d`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 9/20 | 0.450 | 20 | 108–135 |
| extraction | 3/6 | 0.500 | 5 | 66–83 |
| reasoning | 1/2 | 0.500 | 2 | 79–79 |
| relevance | 1/2 | 0.500 | 2 | 90–90 |
| retrieval | 5/6 | 0.833 | 2 | 1910–114809 |
| tool_replay | 12/16 | 0.750 | 8 | 622–715 |
| transcript_qa | 1/2 | 0.500 | 2 | 81–81 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | 10.251 | 22.537 | 41.592 | 1233.067 |
| decode-short | n/a | n/a | n/a | n/a |
| prefill-medium | 11.816 | 11.816 | n/a | 1235.406 |
| prefill-short | n/a | n/a | n/a | n/a |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `code-stable-dedupe:0:42`: score=0.0; status=protocol_error; finish=None.
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
