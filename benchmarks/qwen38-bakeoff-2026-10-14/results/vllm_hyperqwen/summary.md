# Baseline: vllm_hyperqwen

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `a7a533d2e02c5b2b36df4c979e2468db09be2bb606eaa85a0bf97ff48a236149`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 19/20 | 0.950 | 20 | 109–145 |
| extraction | 6/6 | 1.000 | 5 | 64–86 |
| reasoning | 2/2 | 1.000 | 2 | 71–77 |
| relevance | 2/2 | 1.000 | 2 | 85–88 |
| retrieval | 6/6 | 1.000 | 2 | 1911–114812 |
| tool_replay | 16/16 | 1.000 | 8 | 620–715 |
| transcript_qa | 2/2 | 1.000 | 2 | 82–91 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | 12.979 | 18.987 | n/a | n/a |
| decode-short | 0.448 | 6.124 | n/a | n/a |
| prefill-medium | 12.945 | 12.945 | n/a | n/a |
| prefill-short | 0.399 | 0.399 | n/a | n/a |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

- `code-rolling-window:0:42`: score=0.0; status=ok; finish=stop.
