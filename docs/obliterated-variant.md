# OBLITERATED variant: llama.cpp MTP findings

The OBLITERATED Q4_K_M checkpoint runs on one RTX 3090 through the patched
llama.cpp image. Three-token MTP raised median decode from 42.4 to 64.1 tok/s
in the controlled A/B, a +51% change, without a meaningful HumanEval shift or
a concurrency crash. This is a standalone benchmark option. Production still
serves the stock model through vLLM.

## Checkpoint and runtime lane

[`OBLITERATUS/Qwen3.8-27B-OBLITERATED`](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED)
is an Apache-2.0 V3 release based on `Qwen/Qwen3.8-27B`. The publisher calls it
an "abliterated" or uncensored checkpoint, meaning the model was modified to
reduce refusal behavior.

It keeps the stock model's `qwen3_5` hybrid architecture: linear attention
with full attention every 4th layer, 64 layers, an MTP layer at `blk.64`, a
vision tower, and native ctx `262144`. The GGUF architecture string is
`qwen35`.

The tested Q4_K_M file is 16.8 GB. Its sha256 matched the HF-declared LFS oid.
The HF repository ships GGUF and safetensors only. It has no W4A16 checkpoint,
so this option belongs in the llama.cpp lane rather than the vLLM lane.

This work did not compare model quality against stock Qwen3.8-27B. If you use
MMLU figures from the HF card, attribute them to the publisher. This repo did
not measure them.

## MTP flag discovery

Mainline `ghcr.io/ggml-org/llama.cpp:full-cuda` has no `--spec-type` flag at
all. It silently drops the model's MTP weights. Startup logs all 15
`blk.64.*` tensors, including `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`,
and `nextn.shared_head_norm`, as `unused tensor -- ignoring`.

Mainline can therefore never speculate on this checkpoint, and it emits no
error that tells you MTP is unavailable. The locally patched
`llama:cuda-swap-v13-kvarn-rc2` image uses the embedded MTP layer when you pass
`--spec-type draft-mtp`. Its positive startup marker is:

```text
common_speculative_init_result: creating MTP draft context against the target model
```

The patched container has two other launch requirements:

- Set the entrypoint to `/usr/local/bin/llama-server`. Do not pass the
  mainline `--server` argument.
- Pass `--gpus all` even for a `-ngl 0` probe. The binary is CUDA-linked.

The full GPU configuration reached ready in 10s from a cold start.

## Controlled decode A/B

All three arms ran on the patched image with the tiel-bench
`bench_decode_n.py` harness. The harness used production sampling at temp 1.0,
top-k 20, and top-p 0.95. Each arm covered 4 samples across 3 prompt classes,
single stream, ctx 65536, and q8_0 KV.

| Arm | Overall median | Result |
|---|---:|---:|
| MTP off | 42.4 tok/s | baseline |
| `--spec-draft-n-max 3` | 64.1 tok/s | +51% |
| `--spec-draft-n-max 5` | 59.4 tok/s | below n-max 3 |

At parallel 1, draft acceptance was 0.67-0.72. At the shipped parallel 4,
the median was 0.687 over 7 requests, with a 0.578-0.833 range and mean
accepted draft length of 2.73-3.19.

Three draft tokens are the sweet spot. With n-max 5, the model drafts more,
but acceptance falls to 0.56-0.62. The result matches the stock model's k=3
conclusion in the vLLM lane.

Do not compare 64.1 tok/s with the stock model's numbers in this repo or with
the vLLM lane's 95.6 tok/s. The quant, engine, KV type, context, and drafter
all differ. Only the 42.4 to 64.1 tok/s pair is a controlled comparison.

## Correctness control

A 60-item HumanEval subset ran at parallel 4 and concurrency 4:

| Arm | Passed | Score |
|---|---:|---:|
| MTP on | 55/60 | 0.917 |
| MTP off | 54/60 | 0.900 |

The one-item difference at temperature 1.0 is indistinguishable. MTP did not
corrupt output in this control. The run also completed without a crash, which
checks `draft-mtp` concurrency on this stack.

## VRAM limit

The full configuration with vision loaded sits at 22.1 GB of 24 GB. About
2.4 GB of headroom is thin. Concurrent large-image vision encode plus deep
prefills might OOM. That combination was not stress-tested.

## Open leads

The repo's stock profile stacks `--spec-type draft-mtp,ngram-mod` with n-max 5
and a truncated-vocabulary drafter. That combined profile was not tested on
this variant. The n-max 5 arm above used embedded MTP alone.

- `ngram-mod` stacking might help workloads that repeat prompt text, but it
  needs its own controlled measurement here.
- The d48k drafter might reduce draft-head work. Its vocabulary mapping must
  be checked against this checkpoint's 248,320-row head before use.
