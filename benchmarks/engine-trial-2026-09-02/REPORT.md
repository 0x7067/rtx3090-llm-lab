# Benchmark comparison

## decode (single-request, streaming)

| tag | runs | median tok/s | median ttft (s) |
|---|---|---|---|
| llama-dflash2q8-n4 | 5 | 99.61 | 1.16 |
| llama-dflash2q8-n7-ngram | 5 | 291.54 | 1.16 |
| llama-dspark-q8 | 5 | 100.17 | 1.15 |
| llama-mtp-ngram | 5 | 383.03 | 1.09 |
| llama-mtp-ngram-ctrl | 5 | 306.97 | 1.10 |
| llama-mtp-only | 5 | 105.37 | 1.10 |
| sglang-nospec | 3 | 50.18 | 0.79 |
| vllm-prod-v10 | 5 | 128.62 | 0.77 |

## prefill (cold, unique nonced prompt)

| tag | median prompt tokens | median ttft (s) | median tok/s |
|---|---|---|---|
| llama-dflash2q8-n4 | 14767.00 | 13.21 | 1117.50 |
| llama-dflash2q8-n7-ngram | 14767.00 | 13.24 | 1115.30 |
| llama-dspark-q8 | 14765.00 | 13.26 | 1113.80 |
| llama-mtp-ngram | 14767.00 | 12.56 | 1175.40 |
| llama-mtp-ngram-ctrl | 14766.00 | 12.61 | 1170.70 |
| llama-mtp-only | 14766.00 | 12.61 | 1170.60 |
| vllm-prod-v10 | 14769.00 | 12.15 | 1215.10 |

## session (cumulative multi-turn file-editing)

| tag | configs (turns/preamble-tokens) | median cumulative tok/s | median applied rate |
|---|---|---|---|
| llama-dflash2q8-n4 | 20t/20000p, 6t/50000p, 8t/0p | 99.29 | 1.00 |
| llama-dflash2q8-n7-ngram | 20t/20000p, 6t/50000p, 8t/0p | 210.39 | 1.00 |
| llama-dspark-q8 | 20t/20000p, 6t/50000p, 8t/0p | 107.66 | 1.00 |
| llama-mtp-ngram | 20t/20000p, 6t/50000p, 8t/0p | 219.08 | 1.00 |
| llama-mtp-ngram-ctrl | 20t/20000p, 6t/50000p, 8t/0p | 219.16 | 1.00 |
| llama-mtp-only | 20t/20000p, 6t/50000p, 8t/0p | 115.53 | 1.00 |
| sglang-nospec | 6t/0p | 49.42 | 1.00 |
| vllm-prod-v10 | 20t/20000p, 6t/50000p, 8t/0p | 113.69 | 1.00 |

## concurrent (parallel decode)

| tag | n | median aggregate tok/s |
|---|---|---|
| llama-dflash2q8-n4 | 4 | 78.15 |
| llama-dflash2q8-n7-ngram | 4 | 157.74 |
| llama-dspark-q8 | 4 | 78.20 |
| llama-mtp-ngram | 4 | 159.56 |
| llama-mtp-ngram-ctrl | 4 | 157.45 |
| llama-mtp-only | 4 | 82.71 |
| vllm-prod-v10 | 4 | 261.70 |

## sustained (long single generation, windowed decode tok/s)

| tag | median overall tok/s | median first-window tok/s | median last-window tok/s |
|---|---|---|---|
| llama-dflash2q8-n4 | 90.91 | 77.06 | 93.64 |
| llama-dflash2q8-n7-ngram | 112.85 | 103.60 | 117.16 |
| llama-dspark-q8 | 78.86 | 69.33 | 85.37 |
| llama-mtp-ngram | 102.17 | 90.82 | 111.26 |
| llama-mtp-ngram-ctrl | 103.11 | 89.27 | 103.09 |
| llama-mtp-only | 96.78 | 91.85 | 94.15 |
| vllm-prod-v10 | 113.09 | 117.25 | 113.34 |

## quality (4-task battery)

| tag | passed/total | median wall (s) |
|---|---|---|
| llama-dflash2q8-n4 | 4/4 | 1.44 |
| llama-dflash2q8-n7-ngram | 4/4 | 1.22 |
| llama-dspark-q8 | 4/4 | 1.50 |
| llama-mtp-ngram | 4/4 | 1.23 |
| llama-mtp-ngram-ctrl | 4/4 | 1.09 |
| llama-mtp-only | 4/4 | 1.28 |
| vllm-prod-v10 | 4/4 | 1.19 |

