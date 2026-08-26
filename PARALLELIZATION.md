# Serving Tiel to more than one caller at a time

Written 2026-08-26, for the question "if we keep Tiel, how do we parallelize it?"

The deployment currently serves one request at a time. `apps/llama` runs
`--parallel 1`, so a second caller waits for the first to finish. The manifest
explains that as a deliberate trade: llama.cpp divides `--ctx-size` statically
across slots, so any slot count above one cuts the 262k context depth.

That premise is true for the setting the manifest uses, and false in general.
The pinned build has a second mode.

**Recommendation: stay on llama.cpp and add `--kv-unified` with `--parallel 4`.**
Measured numbers are in "What the sweep measured" below. Do not move to vLLM or
SGLang; no checkpoint of this model both fits 24 GB and runs on Ampere.

## Why the runtime cannot change

The 3090 is Ampere, compute capability 8.6. Ornith publishes five artifacts for
Ornith-1.5-35B-A3B — BF16 base, GGUF, FP8, NVFP4, and MLX — and on this card
they fail for different reasons:

| artifact | size | runs on a 3090? |
|---|---|---|
| BF16 base | ~70 GB | No, 3x the card |
| FP8 | ~35 GB | No, still over 24 GB |
| NVFP4 | ~17.5 GB | Fits, but FP4 tensor cores need Blackwell (SM100/SM120). Ampere tops out at INT8 |
| MLX | — | Apple silicon only |
| GGUF Q4_K_S | 20.9 GB | Yes. This is what runs today |

The GGUF is the only artifact that both fits and executes here. That would not
matter if vLLM or SGLang could load a GGUF, but for this architecture they
cannot: vLLM rejects hybrid Gated-DeltaNet GGUFs outright (`GGUF model with
architecture qwen3next is not supported yet`, vllm-project/vllm#30023), and the
same is reported against the Qwen3.5-35B-A3B GGUFs while the safetensors build
of the same model loads fine.

So "move to a paged-KV server" is not a lever here. PagedAttention is the right
answer to this problem in general — it pools KV dynamically instead of
partitioning it — but it needs a checkpoint this card cannot run.

Running two instances is also out: 20.9 GB of weights each on a 24 GB card.

## Why extra slots are cheaper than they look

This is not a plain transformer. The GGUF metadata reports:

```
general.architecture          qwen35moe
qwen35moe.block_count         40
qwen35moe.full_attention_interval  4
qwen35moe.attention.head_count_kv  2
qwen35moe.attention.key_length     256
qwen35moe.attention.value_length   256
qwen35moe.ssm.conv_kernel     4
qwen35moe.ssm.state_size      128
qwen35moe.ssm.inner_size      4096
```

The `ssm.*` keys and `full_attention_interval = 4` mean a 3:1 hybrid: 10 of the
40 layers are full attention and hold a KV cache that grows with context, and
the other 30 are Gated-DeltaNet linear-attention layers holding a **fixed-size
recurrent state per sequence**, independent of how long the conversation gets.

Two consequences:

- **Depth is cheap.** Only 10 layers cache. At `K=q8_0 V=q4_0` that is
  10 x (256 x 2) x (1.0625 + 0.5625) = 8,320 bytes per token, about 2.0 GiB for
  the full 262,144-token context, against roughly 8 GiB if all 40 layers cached.
  That is why the current config fits at full native depth at all.
- **Slots are nearly free.** Each additional sequence costs one more copy of the
  recurrent state — a fixed amount that does not scale with context — plus
  whatever KV its own depth uses.

The report's note that the card lists only 2 KV heads was half the story. The
other half is that three quarters of the layers do not have a KV cache.

## The two llama.cpp modes

From the pinned build's `--help` (image digest `sha256:851b3b87…`):

```
-np,   --parallel N        number of server slots (default: -1, -1 = auto)
-kvu,  --kv-unified, -no-kvu, --no-kv-unified
                           use single unified KV buffer shared across all
                           sequences (default: enabled if number of slots is auto)
-cb,   --cont-batching     whether to enable continuous batching (default: enabled)
```

**Partitioned** (`--no-kv-unified`, which an explicit `--parallel N` selects):
`--ctx-size` is cut into N equal slots. Four slots at `-c 262144` gives each
caller 65,536 tokens, and a single long request cannot borrow from an idle slot.
This is what the manifest describes.

**Unified** (`--kv-unified`): one KV buffer shared by all sequences, isolated by
masking rather than by separate buffers. Slots draw from a common pool, so one
request can still reach the full depth while short requests run beside it. The
constraint becomes the sum: total live tokens across all sequences must fit the
pool, instead of each caller being capped at `ctx/N`.

Unified works on this architecture. llama.cpp routes hybrid models through
`llama_memory_hybrid`, which wraps a `llama_kv_cache` for the attention layers
and a `llama_memory_recurrent` for the Gated-DeltaNet layers and names Qwen3.5
as a supported case. Enabling `kv_unified` sets `n_stream = 1` so all sequences
share one pool of cells, and the hybrid path participates in that; upstream
carries a `test_seq_rm_isolated` test specifically covering sequence removal
under `kv_unified`. So this is a tested combination, not an accident that
happens to load.

Unified is also the better fit for this workload. Coding-agent traffic is bursty
and uneven — one long session against a large repository plus several short
completions — which is exactly the shape that a static partition serves badly.

Continuous batching is already on by default and is independent of slot count;
it is what lets the GPU decode several sequences in one pass once more than one
slot exists.

## What the sweep measured

Measured 2026-08-26 on the idle card, all with the projector loaded and
`K=q8_0 V=q4_0`. Raw data in `results_parallel.json`.

| slots | KV | n_ctx_slot | load | peak | 1 stream | 4 concurrent |
|---|---|---|---|---|---|---|
| 1 (deployed) | partitioned | 262,144 | 23,050 MiB | 23,068 MiB | 112.2 tok/s | 119.1 tok/s |
| 2 | unified | 262,144 | 23,112 MiB | 23,130 MiB | 115.0 tok/s | 162.2 tok/s |
| **4** | **unified** | **262,144** | **23,238 MiB** | **23,258 MiB** | **114.2 tok/s** | **195.3 tok/s** |
| 8 | unified | 262,144 | 23,506 MiB | 23,526 MiB | 113.8 tok/s | 196.4 tok/s |
| 2 | partitioned | 131,072 | 23,112 MiB | 23,130 MiB | 114.6 tok/s | not measurable |
| 4 | partitioned | 65,536 | 23,238 MiB | 23,260 MiB | 115.1 tok/s | not measurable |

Three things fall out.

**Unified KV holds full depth at every slot count.** `n_ctx_slot` stays at
262,144 for 2, 4 and 8 unified slots. The partitioned rows halve and quarter it
exactly as the manifest describes. That is the whole question settled: the
trade the manifest treats as inherent belongs to one flag.

**Slots cost about 63 MiB each.** Load goes 23,050 -> 23,112 -> 23,238 -> 23,506
MiB for 1, 2, 4 and 8 slots. That is the per-sequence recurrent state predicted
above, and it does not grow with context. Four slots cost 188 MiB of the 1,110
MiB the deployment had spare.

**Concurrency scales to about 4.** Aggregate throughput over four concurrent
requests goes 119 -> 162 -> 195 tok/s from 1 to 2 to 4 slots, a 1.64x gain, and
then stops: 8 slots measures 196.4, no better than 4, because the load only ever
offers four requests. Single-stream decode is unchanged throughout (112-115
tok/s), so the extra slots cost an idle caller nothing.

### Why two rows say "not measurable"

The partitioned 4-concurrent cells produced 17.3 and 22.9 tok/s. Those are not
throughput figures. Every configuration decodes ~114 tok/s on one stream, so
even fully serialised execution floors aggregate at ~114; a number seven times
below that is arithmetically impossible for four completed replies.

The harness explains it. `measure()` summed `usage.completion_tokens` from each
response inside a bare `except: pass`, so a response that was not a completion —
an error body, a dropped request — contributed zero tokens against the full wall
clock. The honest reading is that partitioned mode failed to return most of the
concurrent requests, which is a worse result than a slow one, but the run cannot
say how many. `sweep_parallel.sh` now reports a completion count beside every
rate so a re-run distinguishes the two.

This does not affect the recommendation. The `n_ctx_slot` column is the finding,
and it is clean.

### The change

```yaml
- "--parallel"
- "4"
- "--kv-unified"
```

Everything else stays. Expect 23,238 MiB at load against 23,050 MiB today, so
the vision headroom drops from 1,110 MiB to about 920 MiB — still above the
+416 MiB a 3000x2000 image costs at one slot, but that margin has not been
re-measured at four slots and is the thing most likely to break.

## What this does not cover

- Vision. The projector is loaded in every configuration measured here, and its
  peak allocation is per-image work on top of the numbers above. The manifest
  records +416 MiB for a 3000x2000 image at one slot; that has not been
  re-measured at higher slot counts, and it is the most likely cause of an OOM
  after a config change.
- Quality. Slot count does not change weights, but batching changes numerics
  slightly. REPORT.md records about +/-3pp of run-to-run movement on HumanEval
  at n=164, which is wide enough to hide any effect this would have.
