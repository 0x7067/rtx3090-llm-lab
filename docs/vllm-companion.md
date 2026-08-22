# vLLM companion: the current single-3090 deployment

Date: 2026-08-22

This repository remains the llama.cpp optimization project. The current home
deployment is a separate vLLM profile based on
[`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090).
Keeping the engines separate matters: their model formats, caches, speculative
paths, startup behavior, and benchmark harnesses are different. Numbers in
this document are not a direct llama.cpp-versus-vLLM comparison.

## Current profile

| Setting | Value |
|---|---|
| GPU | one RTX 3090, 24 GiB, SM86 |
| Runtime | vLLM 0.27.1 plus the syv-ai patch stack and the local overlay in `../vllm/` |
| Weights | prepared W4A16 compressed-tensors fast variant, 14.83 GiB loaded |
| Vision | enabled |
| Maximum model length | 140,000 |
| KV | FP8 E4M3 through FlashInfer |
| Prefix cache | enabled, Mamba cache mode `align` |
| Recurrent state | FP16 |
| Speculation | built-in MTP, three draft tokens |
| Scheduler | synchronous (`--no-async-scheduling`) |
| Batch budget | 2,048 tokens, eight request slots |
| Tools | `qwen3_coder` tool parser; `qwen3` reasoning parser |

The prepared model is not downloaded from one standalone repository.
[`syvai/qwen3.8-27b-3090-fast-variant`](https://huggingface.co/syvai/qwen3.8-27b-3090-fast-variant)
contains companion files that overlay the prepared W4A16 body. Treating that
repository as a complete checkpoint produces a broken setup.

## Why these defaults won

The matched arm campaign kept MTP-3 with a 2,048-token batch budget:

| Arm | Outcome |
|---|---|
| MTP-3 / 2,048 | promoted control; 143,804-token KV pool |
| MTP-3 / 4,096 | no cold-TTFT win; slower 60k decode; lower C4 aggregate |
| MTP-3 / 8,192 | could not allocate enough KV for the configured 140k length |
| MTP-4 / 2,048 | faster shallow decode, but worse cached depth/concurrency and only 760 tokens of KV margin |

The retained arm measured 104.02 tok/s shallow, 94.50 at 30k, 95.69 at
60k, and 70.41 at 100k. Four concurrent requests produced 358.41 tok/s
aggregate. Cached TTFT was 0.988/1.323/1.780 seconds at 30k/60k/100k, with
58.9% aggregate draft acceptance and no request errors or restarts.

Full evidence:

- [MTP depth and batch-budget arms](https://github.com/0x7067/docker-services/blob/main/docs/research/qwen3.8-mtp-batch-arms-2026-08-22.md)
- [Prefix-cache/speculative investigation](https://github.com/0x7067/docker-services/blob/main/docs/research/vllm-spec-prefix-cache-crash.md)
- [Experiment log](https://github.com/0x7067/docker-services/blob/main/docs/research/vllm-spec-prefix-cache-experiment-log.md)

## Quality position

The compact fast weights report IFBench strict 78.3 versus 79.5 for official
BF16, perplexity 8.095 (about +0.6%), and GSM8K 96.5%. That is a small measured
quantization cost, not proof that every task is unchanged.

No ready-made Hugging Face checkpoint currently proves a quality gain while
also preserving the 140k memory envelope and at least 95% of the control's
speed. The first defensible experiment is a compact rebuild using the Twu31
conversation-calibrated W4A16 body. Pearson's W4A16 checkpoint is the best
fidelity reference but is too large unchanged. The highest-upside behavioral
lead is barozp Opus-Distill-v2, which needs a fresh W4A16 quantization before
it belongs on this GPU.

See the [quality/speed candidate scout](https://github.com/0x7067/docker-services/blob/main/docs/research/qwen3.8-hf-quality-speed-scout-2026-08-22.md)
for evidence, rejects, and the exact A/B gate.

## Reproduce the local v4 overlay

The deployed image was built from syv-ai commit
`b356e31526886b4bd614a79cd8600e7cc9383cf9` plus
[`vllm/syv-ai-b356e315-v4.patch`](../vllm/syv-ai-b356e315-v4.patch).

```bash
git clone https://github.com/syv-ai/qwen38-27b-rtx3090
cd qwen38-27b-rtx3090
git checkout b356e31526886b4bd614a79cd8600e7cc9383cf9
git apply /path/to/qwen38-27b-rtx3090-llamacpp/vllm/syv-ai-b356e315-v4.patch
docker build -t qwen38-27b-3090:v4 .
```

The overlay does three narrow things:

1. changes eight reused FlashInfer CPU planning buffers from pinned to
   pageable memory, with an exact match-count guard;
2. makes vision opt-in through `VISION=1` and permits separate draft attention
   backend/KV overrides;
3. makes the container tree readable when Kubernetes runs it as UID 1000.

The long-context qualified environment is:

```dotenv
CTX=long
VISION=1
MAX_LEN=140000
MAX_SEQS=8
GPU_UTIL=0.95
PREFIX_CACHE=1
DRAFT_TOKENS=3
ASYNC_SCHED=0
```

Do not start it alongside the llama.cpp service or another GPU workload on the
same card. Stop the current owner of the GPU first and wait until VRAM is free.

## Club 3090 local entry

The installable local-layer bundle lives in [`vllm/club-3090/`](../vllm/club-3090/).
It deliberately does not modify Club 3090's curated catalog and does not need
a fork or pull request. The exact Docker Compose wrapper was maintenance-window
qualified on the target 3090 on 2026-08-22 and is now marked `caveats` rather
than `incubating`.

The qualification boot reached `/v1/models` in 108 seconds, served the expected
140,000-token model, stayed healthy with zero restarts, and completed a real
chat request with `finish_reason=stop`. Metrics confirmed FP8 KV, prefix cache,
143,804 KV tokens, and three-token MTP activity. The container used 21,232 MiB
after startup. The existing k3s service was restored and generation-tested
after the canary.

The local slug is:

```text
local/qwen38-27b-single-3090-fast
```

Club's local registry records `drafter: null` because its built-in drafter
compatibility list matches exact core model IDs and rejects a local alias. The
compose itself still runs MTP-3; this is a catalog-metadata limitation, not a
runtime change.
