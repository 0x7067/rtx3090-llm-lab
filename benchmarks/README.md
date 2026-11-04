# Benchmarks

Each campaign keeps its runner, frozen inputs, raw results and decision record
together. Numbers from different campaigns are only comparable when model
bytes, runtime, sampling, context depth and GPU profile all match. Campaign
documents describe the system at their date (see "Reading old documents" in the
[root README](../README.md)).

## Harness

- [`paired-harness/`](paired-harness/): the paired quality and performance
  harness (stdlib Python, OpenAI-compatible API, cluster-bootstrap intervals).
  GitOps qualification scripts call `harness.py`. Do not move it.

## Campaigns, newest first

| Campaign | Question | Outcome |
| --- | --- | --- |
| [`qwen38-bakeoff-2026-10-14/`](qwen38-bakeoff-2026-10-14/) | llama.cpp vs HyperQwen vLLM vs sglang-exl3 vs Strata Flash-Next on base Qwen3.8-27B | sglang-exl3 best overall (54/54); HyperQwen best throughput (269 tok/s at 4 callers). See [REPORT.md](qwen38-bakeoff-2026-10-14/REPORT.md) |
| [`gsq-speculation-2026-10-09/`](gsq-speculation-2026-10-09/) | GSQ with DFlash speculation | Raw runs and diagnosis; no README |
| [`gsq-q4-resume-2026-10-08/`](gsq-q4-resume-2026-10-08/), [`gsq-rco-2026-10-08/`](gsq-rco-2026-10-08/), [`gsq-resident-2026-10-08/`](gsq-resident-2026-10-08/) | GSQ RCO quants (Q4, Q8 resident) on llama.cpp | Raw runs, collectors and `comparison-full.json`; no README. GSQ was not promoted |
| [`orca-thinkingcap-2026-10-07/`](orca-thinkingcap-2026-10-07/) | OrcaSAQ2 speed settings vs ThinkingCap, gated on no quality regression | See `PLAN.md` and the `decision-*.json` / `*-summary.json` files; no written verdict |
| [`local-llm-reset-2026-09-30/`](local-llm-reset-2026-09-30/) | Five-arm baseline after retiring all LLM routes | ThinkingCap on upstream llama.cpp was the only 54/54 and was promoted 2026-10-01 |
| [`orcasaq-vs-w4-2026-09-25/`](orcasaq-vs-w4-2026-09-25/) | OrcaSAQ2 EXL3 vs W4A16 AutoRound | OrcaSAQ2 promoted 2026-09-25; superseded 2026-10-01 |
| [`spark-coding-2026-09-10/`](spark-coding-2026-09-10/) | Spark X2.5 4B coding settings and LoRA prep | Pilot; Spark is no longer served |
| [`qwen-prefill-2026-09-04/`](qwen-prefill-2026-09-04/), [`qwen-speed-2026-09-04/`](qwen-speed-2026-09-04/) | llama.cpp v18 prompt processing and concurrency | +54% on four concurrent coding requests. The `decision.tsv` files are git-ignored |
| [`k2-horizon-2026-09-04/`](k2-horizon-2026-09-04/) | K2 Horizon 7B as an extra backend | Brought up, later retired; weights deleted 2026-11-04 |
| [`engine-trial-2026-09-02/`](engine-trial-2026-09-02/) | llama.cpp master vs vLLM v10 vs SGLang | llama.cpp v14 promoted 2026-09-02. `bench/` is still a useful load tool |
| [`qwen-vllm-hillclimb-2026-08-28/`](qwen-vllm-hillclimb-2026-08-28/) | Apparent vLLM slowdown; 16 optimization arms | No arm cleared the promotion gates |
| [`tiel-vs-qwen/`](tiel-vs-qwen/) | Tiel-Coder-35B-A3B (llama.cpp) vs Qwen3.8-27B (vLLM) deployment fit | Mixed: Tiel faster single-stream and higher on HumanEval, Qwen 6.8x prefill and 2.4x aggregate. Qwen stayed the main model |

HyperQwen's own qualification (2026-10-21 onward) and the November E1-E3
experiments live in the GitOps repo, under `k8s/workloads/apps/llama/`
(`hyperqwen/qualification.json` and `optimization/`).
