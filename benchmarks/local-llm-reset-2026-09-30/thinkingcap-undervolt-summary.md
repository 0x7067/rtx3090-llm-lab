# Baseline: thinkingcap_llama

Single-arm observations; no quality-equivalence or speedup verdict.

Frozen plan: `186ca9720d0c90d16f0d9dd1ca10505bd81c61b4da72f4c5f95ccc3990fe081d`

| Quality tier | Full score / attempted | Mean score | Clusters | Prompt tokens observed |
| --- | ---: | ---: | ---: | ---: |
| coding | 20/20 | 1.000 | 20 | 108–144 |
| extraction | 6/6 | 1.000 | 5 | 64–83 |
| reasoning | 2/2 | 1.000 | 2 | 71–79 |
| relevance | 2/2 | 1.000 | 2 | 89–90 |
| retrieval | 6/6 | 1.000 | 2 | 1910–114809 |
| tool_replay | 16/16 | 1.000 | 8 | 622–715 |
| transcript_qa | 2/2 | 1.000 | 2 | 81–88 |

Mean scores include errors and truncations as zero. Clusters group related cases.

| Performance probe | TTFT (s) | Wall (s) | Server decode (tok/s) | Server prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| decode-medium | 11.983 | 25.108 | 38.934 | 1050.116 |
| decode-short | 0.663 | 12.787 | 42.151 | 771.956 |
| prefill-medium | 13.486 | 13.486 | n/a | 1049.922 |
| prefill-short | 0.673 | 0.673 | n/a | 773.646 |

These are descriptive observations, not a controlled speed comparison.

## Failed or partial-score observations

None.
