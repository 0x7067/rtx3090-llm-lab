# Strata and OrcaSAQ2 speed follow-up — 2026-10-01

## Result

The publisher calibration improved Strata's measured server decode rate. The retained settings are `--pcie-frac 0.00`, `--spec-min-p 0.70`, and `--pool-workers 4`. Across the matched three-seed probes, median server decode rose from 58.0 to 69.0 tok/s on the short prompt and from 43.2 to 63.4 tok/s on the medium prompt. Prefill was effectively unchanged on the short probe and 6% lower on the medium probe. End-to-end medium decode did not improve because host I/O and scheduling dominated its wall time, so the result supports the engine decode settings rather than a general latency claim.

OrcaSAQ2's retained profile is MTP depth 3 with `--max-num-batched-tokens 4096`. Its two decode probes reached a 34.84 tok/s geometric mean, 3.9% above the former depth 2 and 2048-token setting, and 7.2% above depth 1. The 8192-token candidate failed its first measured request with a CUDA allocation failure and is rejected.

## Strata

| Profile | Short server decode | Medium server decode | Short prefill | Medium prefill |
| --- | ---: | ---: | ---: | ---: |
| Control | 58.0 tok/s | 43.2 tok/s | 396.9 tok/s | 1,630.1 tok/s |
| Calibrated | 69.0 tok/s | 63.4 tok/s | 395.1 tok/s | 1,529.9 tok/s |
| Change | +19.0% | +46.8% | -0.4% | -6.1% |

The publisher calibration's internal confirmation was noisy. Its default median was 54.2 tok/s and its retained four-worker profile measured 77.9 tok/s. The independent harness then reproduced the decode improvement over three seeds. The short end-to-end rate improved from 52.17 to 60.87 tok/s; the medium end-to-end rate moved from 26.35 to 25.62 tok/s. No valid run crossed the memory or thermal abort gates.

The supported `enable_thinking=false` coding policy scored 12/15, compared with 6/15 for the thinking control. It still failed two executable behaviors and one extraction-format guard. The simple non-thinking policy scored 2/3 relevance, compared with 3/3 for the control, and is rejected. Strata 0.1.30 ignores `reasoning_budget_tokens`; the attempted 512-token arm was an unchanged control duplicate, and the remaining budget arms were stopped. Quality work ended when the requested priority changed to speed.

## OrcaSAQ2

| MTP depth | Batch tokens | Short E2E decode | Medium E2E decode | Decode geometric mean | Draft acceptance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,048 | 48.14 tok/s | 21.92 tok/s | 32.49 tok/s | 75.3% |
| 2 | 2,048 | 49.90 tok/s | 22.55 tok/s | 33.55 tok/s | 65.4% |
| 3 | 2,048 | 51.92 tok/s | 22.79 tok/s | 34.40 tok/s | 50.9% |
| 3 | 4,096 | **51.92 tok/s** | **23.38 tok/s** | **34.84 tok/s** | 50.4% |

Each row is the median of three fixed seeds under the same image, checkpoint, GPU profile, 150,000-token context, FP8 KV cache, prefix cache, eight-sequence limit, and request plan `309ac0d2620b6c32beaa558305ec4dac0027d0e49d3d01ef4ac146e93cc23be0`. The rates include TTFT because vLLM's OpenAI response did not expose server decode timing.

Depth 3 with 8192 batch tokens started, but the first request failed in `solve_tril`: the runtime tried to allocate 48 MiB with 52 MiB free. Its generated summary contains error-derived values and must not be used for comparison. Depth 3 with 4096 batch tokens completed without a monitor abort or thermal trip.

## Deployment

The production candidate uses image digest `sha256:37ae52afe6b11ffa4b613f1fa5ecd8354584a32daa500b7ce02dbdb1c283e012`, which is the exact local image used for the benchmark. It serves `qwen3.8-27b` through vLLM 0.30.0 and OrcaSAQ2 kernel 0.1.0 with MTP depth 3, 4096 batch tokens, 150,000 context, eight sequences, FP8 KV, and prefix caching.

## Evidence

- `strata-publisher-calibration.json` records the publisher calibration sweep and retained settings.
- `strata-speed-control-*` and `strata-speed-calibrated-*` contain the matched three-seed runs, summaries, logs, and telemetry.
- `orca-speed-mtp{1,2,3}-mbt2048-*` and `orca-speed-mtp3-mbt4096-*` contain the valid OrcaSAQ2 runs.
- `orca-speed-mtp3-mbt8192-server.log` retains the rejected candidate's CUDA failure.
- `speed-plans/plan-speed.json` freezes the request plan.
