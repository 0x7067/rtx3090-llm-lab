---
title: llama
---

# llama

llama-swap endpoint on the single RTX 3090 of `denguinho-server`, in the `apps`
namespace. Flux reconciles this folder. The manifests are the source of truth;
this README only explains how they fit together.

## Source of truth

| File | Owns |
| --- | --- |
| `swap-configmap.yaml` | The `llama-swap-config` ConfigMap: the model list, aliases, TTLs, preload, per-model commands, env and capabilities. Every model id comes from here. |
| `deployment.yaml` | The pod: init and sidecar containers, image digests, probes, resources and hostPath mounts. |
| `service.yaml`, `networkpolicy.yaml`, `podmonitor.yaml` | Service, network policy and metrics scrape. |
| `kustomization.yaml` | Resources, plus the `llama-chat-template` and `llama-vllm-metrics-proxy` ConfigMaps built from `chat_template.jinja`, `chat_template_uncensored.jinja`, `start_qwen_cache.py`, `serve_observer.py` and `vllm-metrics-proxy.py`. |

## Containers

- `gpu-profile` (init) sets the GPU power limit and restores stock clocks before serving.
  Its values and rationale are in `deployment.yaml`. Its image `orcasaq2-vllm` is
  pinned by digest; this repo has no build recipe for it.
- `llama-swap` runs `/usr/local/bin/llama-swap` with the config mounted at `/config`; its flags are in `deployment.yaml`.
  Its image packages the private SparseKV runtime recorded in `deployment.yaml`.
  Qualification and rollback evidence are in `image/sparsekv/`.
  Probes use `/health`.
- `vllm-metrics` (sidecar) runs `vllm-metrics-proxy.py`, a read-only proxy for the
  HyperQwen vLLM `/metrics` with observer status gauges. Its image is `home-llm-strata:0.2.8-v1`.

The base image is built by hand with `image/strata/build.sh [tag]`. It pushes to
the local registry (see the `build-image` skill) and prints a digest reference to paste into `deployment.yaml`.
Bump the tag on every rebuild; containerd caches by tag.

## Models

The chat catalog serves two models on the RTX 3090:

- `qwen3.8-pi` is preloaded and uses the original harness-hub LoRA, MTP4,
  172,032 tokens of context, and the qualified PTH prefix cache. See
  [PI-ADAPTER](PI-ADAPTER.md).
- `inkling-2-mini-unbound` loads on demand. See
  [Inkling 2 Mini on demand](#inkling-2-mini-on-demand).

llama-swap unloads the current worker when a request selects a different model.
Both chat workers stay resident until switched, so repeated requests can reuse
the prompt cache. The private image-generation and OCR routes remain available
to their applications and can also take the GPU when called. See
[`qwen-image/`](qwen-image/README.md) and [`ocr/`](ocr/README.md).

## Client metadata in /v1/models

Entries set `capabilities` (`in`, `out`, `disableAuto`) and/or `metadata`
(`max_model_len`, `max_tokens`, `reasoning`, `input`) in `swap-configmap.yaml`.
llama-swap echoes these in the `/v1/models` listing, so check the live listing
rather than copying field names elsewhere.

## Inkling 2 Mini on demand

Select `inkling-2-mini-unbound` on the authenticated llama endpoint.
The [Ampere vLLM port](https://github.com/Hiutaky/vllm/tree/6e9d1cce3eadb9844d491de9b65dba24b9fec566)
serves a **262,144-token (256K) context** with 512-token prefill chunks and one
inline image per prompt. Send image attachments as base64 `data:` URLs.
`inkling-2-mini-unbound` shares the same read-only NVMe checkpoint and applies the
803 KB `nokeepai/Inkling-2-Mini-Unbound` `abliteration/edit.pt` (20 BF16 `wo_ud`
tensors, SHA-256 verified in memory on load via `INKLING_UNBOUND_EDIT`).
The RTX 3090 profile reserves 5.55 GiB for its FP8 KV cache (267,888 tokens,
`1.02x` of 256K, leaving ~970 MiB total NVIDIA driver headroom for on-demand
Humming cubin loads), offloads input embeddings, the vision encoder, and
DSpark's `markov_w1` embedding table to CPU via UVA
(`--cpu-offload-params embed_tokens visual markov_w1`), serves up to 8
concurrent requests (`--max-num-seqs 8`, selective decode CUDA graphs
`[1,2,4,8]`, adaptive `--long-prefill-token-threshold 256`,
`--cache-aware-admission-window 8`), runs probabilistic DSpark speculative
decoding (`num_speculative_tokens=7`), and loads the `inkling-sm86-opt/`
ConfigMap overlay (unscaled uint16 FP8-E4M3 global + local SWA-1024 attention
with `BN=64` global decode across `N=1..16`, `sp=1` fused local decode,
`sp=20/10/5` concurrency-tuned global splits, and `sp=2` global prefill;
SM86 low-M BF16 GEMV for `qkvr_local` with cuBLAS `wo_ud`; 0-padding GCD KV
group sizing; FP8 DSpark draft KV + zero-VRAM on-the-fly RoPE + L2-swizzled
concurrency-tuned per-row INT8 W8A16 draft/Markov/LM-head projections + fused
`markov_fold + add_softcap`; and SM86 autotuned deterministic Humming expert
schedules). The Inkling parsers handle reasoning and tool calls. llama-swap
unloads the current worker before loading another model. Qwen Pi remains the
startup preload.

Use the authenticated Tailnet URL `http://100.64.0.2/v1` for long requests.
The public Cloudflare route has a
[125-second response timeout](../../../../host/clef/README.md).

Stage the pinned runtime and verify the weight files on the llama host before
merging a change to these mounts:

```bash
k8s/workloads/apps/llama/prepare-inkling-vllm.sh
```

The staging script extracts the Python and CUDA tree from the pinned vLLM
image and records its digest, the port's commit, and the runtime file hashes.
The pod mounts this bundle read-only from Buttercup and uses the host's NVIDIA
driver. Staging reads the image from the Docker store on Buttercup. The
checkpoint is verified in a standalone directory on the second NVMe before
publication and mounted read-only. The launcher keeps Inkling's compiler
and application caches under `/cache/inkling-vllm`.

The NInfer-all 2026-11-21.2 bundle stays mounted for a GitOps rollback.
Revert the Inkling vLLM change and reconcile the `apps` Flux Kustomization to
restore NInfer. To rebuild the rollback bundle, run
`k8s/workloads/apps/llama/prepare-inkling-2-mini.sh` on the llama host.

## Qualification records

Qualification records are indexed in [optimization/README.md](optimization/README.md).
They are evidence, not configuration: change behaviour in `swap-configmap.yaml`.

## Prefix caching

Prefix caching is configured per model in `swap-configmap.yaml`. Clients keep
prefixes cacheable by:

- keeping system prompts, tool definitions and tool ordering stable;
- serializing tool schemas deterministically;
- putting timestamps and request-specific data after the reusable prefix;
- appending messages instead of rebuilding earlier prompt sections;
- keeping reasoning and chat-template options stable within a session.

The HyperQwen SparseKV model uses explicit 864-token cache blocks, matching
its installed kernels. Its optional saved-prefix warmer and private cache
bundle are described in [OBSERVABILITY](OBSERVABILITY.md#saved-prefix-warming).
The [prefix/MTP qualification](optimization/prefix-mtp/README.md) records the
measured client cache hits and draft-count comparison.

## Storage

Read-only serving inputs are mounted from the root NVMe under
`/srv/llama-nvme/production-models`. This includes the configured text models,
Qwen-Image, and the OCR model and runtime directories. The OCR job queue keeps
its durable Buttercup path. Inkling uses the second NVMe to preserve space
for etcd on the root SSD.

Before deploying these mounts, run `prepare-ssd-models.sh` as root. It copies
the configured assets from Buttercup, verifies each rsync transfer and checks
for metadata differences. The pod requires its completion marker at
`/srv/llama-nvme/production-models/.staged-at`. An init container validates the
Finance OCR verification record using the same mounts and identity as the
launcher. If the record is stale after a move, the pinned preflight checks the
artifact contents and recreates the record before the service starts. The
archived originals remain
in the model library at the paths documented in
[`docs/model-storage.md`](../../../../docs/model-storage.md).

## Deploy

After pushing to `main`:

```bash
flux reconcile kustomization apps --with-source
kubectl -n apps logs deploy/llama -c gpu-profile
kubectl -n apps logs deploy/llama -c llama-swap -f
```

Validate a `swap-configmap.yaml` edit by rendering the manifests locally, then
run the packaged `llama-swap -validate` in a CPU-only validation container.
Check llama-swap's log after reconciliation and qualify the authenticated API.

```bash
kubectl kustomize k8s/workloads/apps/llama >/dev/null
```

## Test

Local:

```bash
export LLAMA_API_TOKEN="$(sops --decrypt --extract '["stringData"]["LLAMA_API_TOKEN"]' k8s/workloads/apps/llama/api-secret.enc.yaml)"
kubectl -n apps port-forward svc/llama <local-port>:http
curl -s http://127.0.0.1:<local-port>/v1/models \
  -H "Authorization: Bearer ${LLAMA_API_TOKEN}" | jq '.data[].id'

curl http://127.0.0.1:<local-port>/v1/chat/completions \
  -H "Authorization: Bearer ${LLAMA_API_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-hyperqwen","messages":[{"role":"user","content":"Write a tiny Python fibonacci function."}],"max_tokens":128}'
```

Remote clients use the authenticated Caddy Tailnet route while connected to the Tailnet:

```bash
TOKEN="$(sops --decrypt --extract '["stringData"]["LLAMA_API_TOKEN"]' k8s/workloads/ingress/caddy/secret.enc.yaml)"
curl -s http://100.64.0.2/v1/models \
  -H "Authorization: Bearer ${TOKEN}" | jq '.data[].id'
```

Benchmark a live model with `scripts/benchmark-llama.sh`. Its header lists the
environment variables; `MODEL` defaults to `qwen3.8-27b-hyperqwen`:

```bash
MODEL=qwen3.8-27b-hyperqwen RUNS=3 MAX_TOKENS=256 scripts/benchmark-llama.sh
```

## Tuning and troubleshooting

- Edit the model's entry in `swap-configmap.yaml`, validate it as above, then push.
- If a model runs out of GPU memory on the 24 GB card, lower its context size before
  raising KV precision.
- If the pod does not become ready, read `llama-swap` logs. If `gpu-profile` fails,
  read its log: its script asserts the GPC and memory offsets it sets.
- A model that fails to boot with missing weights is most likely the uncensored entry
  (see its section above).

## CUDA hang capture

The archived SparseKV launcher can record bounded CUDA trace windows from process
startup. The restored DFlash production profile uses its original launcher. See [CUDA-TRACING.md](CUDA-TRACING.md) for retention, capture state,
manual evidence preservation, and disabling the recorder.

## Cyber fine-tune (qwen3.8-27b-cyber)

Artefact B of the 2026-11 local fine-tune: the abliterated Qwen3.8-27B
(`JonathanColetti/Qwen3.8-27B-Uncensored`) QLoRA-trained for 400 steps on the
Vyber07 cyber-security corpus, merged to bf16 and quantised to Q4_K_M
(16.8 GB). Valid-loss 0.3499; 0 refusals in 100 harmful prompts; the 30-prompt
cyber/general suite shows no planning preamble (thinking is disabled).

- Weights: `/srv/llama-nvme/hyperqwen-bakeoff/models/qwen3.8-27b-cyber/model.gguf`
  (archived legacy artifact; the original host path is retained as a symlink).
  See the [storage map](../../../../docs/model-storage.md#november-18-legacy-serving-archive).
- Runtime: a bundled CUDA build of upstream llama.cpp (`3549c27`, arch 86, NCCL
  off) under `.../qwen3.8-27b-cyber/runtime/`, because the image's own
  `/opt/llama/llama-server` is CPU-only and reported "no usable GPU found".
  The binary needs `libcuda.so.1` from `/usr/local/nvidia/lib64`; its CUDA 12
  runtime (cudart/cublas/cublasLt) is bundled next to it.
- Consequence for the abliterated sibling: Heretic's export dropped the MTP
  draft layer, so a GGUF converted from that bf16 model has 851 tensors where
  the loader expects 866 and will not load. Artefact A therefore exists only as
  bf16 HF weights; only artefact B is servable as GGUF.

### Thinking policy for `qwen3.8-27b-cyber`

Served with reasoning **on and bounded** (`--reasoning-format deepseek
--reasoning-budget 512`), not suppressed. Measured on the 30-prompt suite with
identical weights and server flags:

| Arm | tokens/answer | latency/answer | planning text in `content` | refusals |
| --- | --- | --- | --- | --- |
| thinking off | 422 | 10.6 s | 0 | 1/100 |
| thinking on, budget 512 | 1149 | 28.7 s | 0 | 1/50 |
| thinking on, unbounded | 2573 (cap) | 64 s | 0 | n/a |

Unbounded thinking hit the 3,000-token cap with empty `content` on 3 of 5
prompts, so the budget is what makes reasoning usable here. With the budget the
visible answer is clean either way — `--reasoning-format deepseek` puts the
reasoning in `reasoning_content`, which is why the earlier "planning leak"
needed neither suppression nor a retrain to disappear.

Per-request control works: a request carrying
`chat_template_kwargs: {"enable_thinking": false}` answers in 6.5 s with no
reasoning field, while the default path spends ~4x. Per-request
`reasoning_effort` is only partly usable — `low` is accepted, `high` makes the
Qwen template raise a 500 — so no effort selector is exposed in DSH.

The corpus holds essentially no reasoning traces: a full scan of all 1,927,910
rows found `<think>` tags in 2,610 (0.14%), "reasoning effort" in none, and
planning meta-text at the start of an assistant turn in 185 (0.01%). A
reasoning-aware fine-tune would therefore need new data, not another 400-step
run on this corpus.

### Served quantisation

`IQ4_XS` (15.41 GB), not `Q4_K_M`. Same flags, same prompts, `llama-bench -r 3`
on the box:

| file | pp8k | pp34k | tg512 | 128k VRAM | gates |
| --- | --- | --- | --- | --- | --- |
| Q4_K_M (16.8 GB) | 1475.9 | 1293.2 | 40.74 | 21.4 GB | 0 leaks, 1/50 refusals, 28.7 s |
| IQ4_XS (15.41 GB) | 1628.8 | 1421.1 | 45.14 | 20.1 GB | 0 leaks, 0/50 refusals, 26.4 s |

Rejected by measurement, not by preference: `-ub 4096/-b 8192` (−1.4% pp8k),
`-ub 1024` (−2.7%), `-t 8`/`-t 16` (−1.0%, inside noise), f16 KV at 32k (+0.6%/±1.4%,
inside noise), `-fa off` with q8 KV (aborts: "quantized V cache requires flash_attn
to be enabled" — FA is engaged, not assumed), Q5_K_M (19.5 GB + q8 KV at 128k
does not fit the card), MTP (llama.cpp logs the `blk.64.*` tensors as unused).
