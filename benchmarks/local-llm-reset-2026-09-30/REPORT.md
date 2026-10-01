# Local LLM reset baseline — 2026-10-01

## Result

The former HyperQwen, llama, and Qwen DFlash serving resources and home/VPS LLM routes were retired in GitOps commit `f442ea4`. Model PVCs and local weight files remain. The baseline ran in isolated Docker containers bound to loopback; every trial container is now stopped and the host GPU profile has been restored.

ThinkingCap on upstream llama.cpp was the only arm to score 54/54. GSQ on the same runtime scored 53/54. The two vLLM arms each scored 52/54 for different reasons: W4A16 rejected two deep requests at context admission, while OrcaSAQ2 admitted every request and missed two executable coding checks. Strata scored 48.67/54. These are observed results from one authored synthetic run per arm, not statistical proof that one model is generally better.

| Arm | Runtime | Quality score | Full passes | Main misses |
| --- | --- | ---: | ---: | --- |
| ThinkingCap Q4_K_S | llama.cpp b11277 | **54/54** | **54** | None |
| ISTA GSQ/RCO IQ3_S | llama.cpp b11277 | 53/54 | 53 | `code-sse-frames` |
| dbirks W4A16 AutoRound | vLLM 0.30.0 | 52/54 | 52 | Two 114K retrieval requests rejected by the 98,304-token context limit |
| OrcaSAQ2 EXL3 3.21 bpw | vLLM 0.30.0 plus OrcaSAQ2 kernel | 52/54 | 52 | `code-sse-frames`, `code-histogram` |
| Qwen3.8 Flash Next 125B IQ2_XS | Strata 0.1.30 | 48.67/54 | 48 | Three coding failures, two extraction failures, one partial relevance score |

All arms passed 16/16 tool replay cases. ThinkingCap, GSQ, OrcaSAQ2, and Strata passed all six retrieval cases through 114,811 observed prompt tokens. W4A16 passed the four retrieval cases through 28,791 observed prompt tokens, then returned HTTP 400 for both 114K cases because 81,921 or more input tokens plus the requested 16,384 output tokens exceeded its configured 98,304-token window.

## Frozen conditions

The final plan SHA is `7037bfaa64ef760a7c690c59d4f6347849b8a2e208eca122eb860be24babb0cb`. It contains 54 scored quality cases and four performance probes. Quality requests use temperature 1.0, top-p 0.95, top-k 20, low reasoning effort for coding, medium otherwise, and a common 16,384-token output cap. Performance probes retain their authored one-token or 512-token limits. The suite is an authored infrastructure regression set, not a representative sample of production tasks.

The Python grader image is `sha256:9e87977b867847e186d066f531ef783b006d582a985c341c269446088d90f2c4`. The engine images are:

- upstream llama.cpp b11277, commit `eae11d221`, image digest `sha256:7149a45c80596644320de1d565db1b3b8e70b22c66fed612e69b73f6b1e0d746`;
- vLLM 0.30.0 with `orcasaq2-kernel==0.1.0` and `exllamav3==1.5.1`, local image ID `sha256:1d60e0a2fc9eb37042f94b8d5fb23457df6a27ca365305854b1bbcea8b4d9308`;
- Strata 0.1.30 from publisher commit `30ec18ec7094550fcc594fd948220d511d80464e`, local image ID `sha256:f0aead168c875cb4035ad7e8647c015939658866e7f37089bb2a126f2f06dda4`.

The GPU was an RTX 3090 under driver 595.91.07. Matched runs used a 210–1350 MHz graphics lock, +100 graphics VF offset, stock memory VF offset, 370 W power limit, the existing memory-aware fan curve, and the 101°C VRAM trip guard. No final run tripped the guard. The normal profile was restored afterward: graphics VF offset 0, memory VF offset 750, `gpu-mem-oc`, the fan controller, and the thermal guard active.

## Runtime compatibility

| Artifact | vLLM | llama.cpp | Strata |
| --- | --- | --- | --- |
| OrcaSAQ2 EXL3 3.21 bpw | Measured | Exact artifact incompatible | Not applicable |
| dbirks W4A16 compressed tensors | Measured | Exact artifact incompatible | Not applicable |
| ThinkingCap Q4_K_S GGUF | Skipped by request | Measured | Not applicable |
| ISTA GSQ/RCO IQ3_S GGUF | Skipped by request | Measured | Not applicable |
| Flash Next 125B IQ2_XS shards | Not applicable | Not tested as a different engine/model path | Measured |

GGUF under vLLM was intentionally skipped. The EXL3 and compressed-tensors checkpoints are not readable by llama.cpp, so those exact matrix cells are marked incompatible rather than substituted with different quantizations.

Strata is a separate fifth arm, not another quant of the 27B model. It used the original `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` 125B IQ2_XS shards at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, with both publisher SHA256 values verified. Its publisher configuration used 150,000 context, int8 KV, text only, low-RAM resident experts, KV streaming, and a 17.56 GiB GPU expert cache.

## Performance probes

Each cell is one observation. `E2E tok/s` includes TTFT; `server tok/s` is the engine-reported decode rate and is unavailable from these vLLM OpenAI responses.

| Arm | Medium TTFT | Medium E2E tok/s | Medium server tok/s | Short TTFT | Short E2E tok/s | Short server tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ThinkingCap | 11.972 s | 20.41 | 38.95 | 0.649 s | 40.17 | 42.24 |
| GSQ/RCO | 12.008 s | 19.75 | 36.73 | 0.659 s | 37.84 | 39.70 |
| W4A16 | 12.779 s | 20.71 | n/a | **0.428 s** | 42.23 | n/a |
| OrcaSAQ2 | 12.867 s | 21.67 | n/a | 0.571 s | **49.74** | n/a |
| Strata | **9.935 s** | **25.29** | **49.60** | 1.110 s | 44.36 | 49.00 |

Medium and short prefill-only observations:

| Arm | Medium TTFT | Medium server prefill | Short TTFT | Short server prefill |
| --- | ---: | ---: | ---: | ---: |
| ThinkingCap | 13.424 s | 1050 tok/s | 0.651 s | 775 tok/s |
| GSQ/RCO | 13.428 s | 1047 tok/s | 0.699 s | 750 tok/s |
| W4A16 | 12.776 s | n/a | **0.400 s** | n/a |
| OrcaSAQ2 | 12.810 s | n/a | 0.548 s | n/a |
| Strata | **8.069 s** | **1531 tok/s** | 1.117 s | 349 tok/s |

The harness generated paired comparison JSON for ThinkingCap against every other arm and for W4A16 against OrcaSAQ2. Per-tier bootstrap verdicts remain exploratory or inconclusive because the suite has one repeat, few clusters in most tiers, and no multiplicity adjustment. Identical request seeds do not produce identical random streams across engines.

## Effect of the corrected token cap

The earlier 4,096-token Strata run scored 43/54 and is retained only as diagnostic history. Under 16,384, seven former failures passed: `retrieve-join-4096`, `code-range-coalesce`, `code-sse-frames`, `code-histogram`, `code-flatten`, `code-rate-limit`, and `code-topological`.

Two cases still exhausted 16,384 tokens without a usable answer: `code-diff-keys` and `code-merge-config`. `code-redact` reached a natural stop but failed its executable test. `extract-missing` remained wrong; `extract-quoted` became a new miss, and `support-relevant-db` scored two-thirds. Because the request plan changed and sampling is stochastic, the 4,096 and 16,384 totals are diagnostic rather than a formal paired quality comparison.

## Resource and thermal envelope

| Arm | Harness time | Max GPU memory | Max container RAM | Min host available | Max PSI avg10 | Core / hotspot / VRAM max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ThinkingCap | 16.3 min | 20,454 MiB | 8.85 GiB | 39.80 GiB | 0.00 | 64 / 78 / 94°C |
| GSQ/RCO | 19.8 min | 17,314 MiB | 20.31 GiB | 39.25 GiB | 0.00 | 69 / 82 / 90°C |
| W4A16 | 11.2 min | 21,938 MiB | 16.24 GiB | 42.67 GiB | 0.18 | 63 / 75 / 90°C |
| OrcaSAQ2 | 15.8 min | 23,268 MiB | 11.60 GiB | 42.56 GiB | 0.00 | 71 / 83 / 86°C |
| Strata | 44.3 min | 23,850 MiB | 33.48 GiB | 25.95 GiB | 2.61 | 66 / 79 / 82°C |

The abort gates were 12 GiB host memory available and memory PSI `some` avg10 no higher than 5. Every arm stayed inside both. ThinkingCap's root temperature logger began after request 12 because root sensor access was enabled during the run; its temperature maxima therefore cover the remainder rather than the entire final run. Host/GPU telemetry covers all measured requests. The independent one-second thermal guard covered the full campaign.

An earlier, pre-profile ThinkingCap attempt hit 102°C VRAM and was killed by the host guard. Its partial output remains separate and is not mixed into these results.

## Artifacts

- `plan-16384.json`, `suite-post-upgrade.json`, and `arms-post-upgrade.json` freeze requests and provenance.
- `comparison-16384.json` is the compact five-arm quality, performance, and resource matrix.
- `*-16384-run/` contains each manifest, excluded warmup, and raw response JSONL.
- `*-16384-summary.json` and `*-16384-summary.md` contain single-arm summaries.
- `compare-*-16384.json` contains paired analyses.
- `*-16384-telemetry.tsv`, `*-16384-thermal.jsonl`, and `*-16384-server.log` retain host, root sensor, and engine evidence.
- `run-remaining-16384.sh`, `thermal-logger-16384.sh`, and `aggregate-16384.py` reproduce orchestration and aggregation.

The practical result of this bounded suite is clear: ThinkingCap is the cleanest baseline; GSQ is close but missed one coding behavior; W4A16 is strong inside its usable context range but cannot satisfy the frozen deep requests with a 16K output reservation; OrcaSAQ2 covers the full context but missed two coding checks; Strata is fast for a 125B resident-expert setup and benefits materially from the larger output budget, but remains less reliable on this suite.
