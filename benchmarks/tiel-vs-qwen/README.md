# Tiel-Coder-35B-A3B vs qwen3.8-27B on one RTX 3090

A deployment-fit comparison of two local coding models on a single RTX 3090
(24 GB, Ampere cc 8.6), plus the harnesses that produced it. Tiel-Coder-35B-A3B
(Ornith-1.5-35B-A3B, hybrid Gated-DeltaNet MoE) runs on llama.cpp; qwen3.8-27B
runs on vLLM. Both are measured at the configuration each actually served in,
not at a shape chosen to benchmark well.

This directory preserves the former `tiel-bench-rtx3090` public repository and
its original Git history. The consolidated copy also includes later benchmark
scripts and results from the private qualification checkout. General Qwen vLLM
runtime research lives in [`../../research/vllm/`](../../research/vllm/).

**The interesting result is not the winner. It is that the first version of this
benchmark was wrong in two independent ways, and the harnesses could not prove
it.** Both defects are documented in `REPORT.md` along with the fixes.

## Headline

| metric | qwen3.8-27b | Tiel-35B-A3B | winner |
|---|---|---|---|
| Decode, single stream (tok/s) | 111.2 | **121.8** | Tiel, 1.10x |
| Prefill, ~6.8k uncached (tok/s) | **1269** | 188 | **qwen, 6.8x** |
| Aggregate, 4 concurrent (tok/s) | **278** | 114 | qwen, 2.4x |
| HumanEval pass@1 | 89.0% | **96.3%** | Tiel, +7.3pp |
| Multi-turn repair, 3 turns | 96.0% | **100%** | Tiel |
| Replies containing no code (of 164) | 15 | **1** | Tiel |
| MMLU-Pro, excl. truncated | 87.7% | 87.0% | tied |

## Three things worth reading the report for

**A token limit was measuring formatting as quality.** Both quality harnesses
sent `max_tokens=6144`. A reply that ran out of budget arrived with no fenced
code block, and an empty extraction scores as a failure. The two models hit that
wall at very different rates, so the original pass@1 and multi-turn rows partly
measured truncation. Raising the limit to 12,288 moved qwen from 85.4% to 89.0%
and Tiel from 92.7% to 96.3%.

**Every wrong fix in the repair benchmark belonged to one model — the one that
looked worse.** Split the repair-loop replies by whether they contained code at
all and qwen was correct on 120 of 120 submissions, at both token limits. It has
never proposed a wrong fix in this test. Its five unsolved tasks answered with
prose on all three turns, so no candidate was ever executed for them. The
reported 96.0% vs 100% is five tasks that produced no code, not five wrong fixes.

**Prefill collapsed on the shipped configuration.** The first run measured Tiel
prefilling at 2805 tok/s, a 2.16x win, on a 4-slot 16k f16-KV shape. The config
that actually shipped — 262k context, one slot, `K=q8_0 V=q4_0`, vision
projector — prefills the same prompt at 188 tok/s. That is 36.7 seconds to first
token against qwen's 5.3. For agents working against a large context this
dominates everything else in the table, and the cause is not yet isolated.

## Parallelizing llama.cpp on this model

`PARALLELIZATION.md` answers a separate question: how to serve more than one
caller when only one GPU is available.

llama.cpp splits `--ctx-size` statically across `--parallel` slots, so more slots
normally means less depth each. That is true only on the `--no-kv-unified` path.
With `--kv-unified` the cache is one buffer shared by all sequences, and a
measured sweep found `n_ctx_slot` staying at the full 262,144 for 2, 4 and 8
slots. Four slots cost 188 MiB and raise four-way aggregate throughput from 119
to 195 tok/s with single-stream decode unchanged.

Extra slots are nearly free here because of the architecture. The GGUF reports
`full_attention_interval = 4` alongside `ssm.*` keys: 10 of 40 layers hold a
growing KV cache, the other 30 hold a fixed-size recurrent state per sequence.
Full 262k depth costs about 2 GiB rather than 8, and each additional sequence
adds a constant that does not scale with context.

Moving to vLLM or SGLang for PagedAttention is not available on this card. Of the
five artifacts published for this model, BF16 (~70 GB) and FP8 (~35 GB) exceed
24 GB, NVFP4 fits but needs Blackwell FP4 tensor cores that Ampere lacks, MLX is
Apple-only, and vLLM rejects hybrid Gated-DeltaNet GGUFs outright.

## Harnesses

| file | what it measures |
|---|---|
| `bench_speed.py` | TTFT, decode, uncached prefill, 4-way concurrency. A unique random preamble per request defeats prefix caching on both stacks |
| `bench_quality.py` | HumanEval pass@1, greedy, one sample per problem |
| `bench_multiturn.py` | repair loop over seeded bugs, re-executing between turns and feeding back real stderr |
| `bench_mmlu.py` | MMLU-Pro, stratified fixed-seed sample |
| `make_mutants.py` | deterministic bug seeding; only mutants verified to actually fail are kept |
| `sandbox_runner.py`, `batch_exec.py` | execute model-generated code in a network-less container |
| `sweep_parallel.sh` | the slots-by-depth frontier for llama.cpp |
| `compare_v2.py` | prints both token limits side by side, with no-code counts attributed to a `finish_reason` |
| `test_no_code_accounting.py` | checks that a reply containing no code is accounted for rather than silently scored as wrong |

Both quality harnesses persist the full reply text and `finish_reason`. That is
the change that made the second run diagnosable and the first one not: the
original stored only extracted code, so a truncated reply and a wrong answer were
indistinguishable afterwards.

All generated code runs inside a network-less container. Model weights and the
MMLU-Pro parquet are gitignored; both are re-fetchable.

## Running it

Endpoints and paths come from the environment:

```bash
export LLAMA_URL=http://127.0.0.1:8080        # OpenAI-compatible endpoint
export MODELS_DIR=$PWD/models                 # GGUF weights live here

uv run bench_quality.py "$LLAMA_URL" my-model HumanEval.jsonl cand.jsonl 2
uv run bench_speed.py   "$LLAMA_URL" my-model results_speed.json
uv run compare_v2.py
```

The orchestration scripts (`bench_window.sh`, `run_qwen.sh`, `restore_qwen.sh`)
assume a Kubernetes deployment named `llama` in an `apps` namespace reconciled by
Flux. They are included because they document the exact sequence used, including
the ordering constraint that cost the most to rediscover: Flux must be suspended
*before* scaling the deployment to zero, or the next reconcile puts the model
back on the card beside whatever the benchmark started.

## Caveats

These are two deployments on one card, not two architectures in isolation.
Neither HumanEval nor MMLU-Pro predicts performance on your codebase; HumanEval
in particular is saturated at this level. Run-to-run movement is about ±3pp on
HumanEval, and much larger on qwen's decode, which uses MTP speculative decoding
and varies with draft acceptance. Vision is deployed but untested here.

`REPORT.md` states which numbers are load-bearing and which are not.
