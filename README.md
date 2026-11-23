# RTX 3090 LLM Lab

Evidence, build sources and tooling for serving local LLMs on one 24 GB RTX 3090
(Ampere, sm_86) in the home k3s cluster. This repo is the `rtx3090-llm-lab`
submodule of the GitOps repo `docker-services`, which holds the manifests Flux
applies.

**Which repo answers what:**

- *What is running, and with which flags?* [`production/STATUS.md`](production/STATUS.md),
  generated hourly from GitOps `k8s/workloads/apps/llama/`. GitOps stays the
  source of truth; [`production/gitops/`](production/gitops/) is a read-only
  copy pinned in [`production/LOCK.json`](production/LOCK.json).
- *Why was it chosen, and how was it measured?* This repo: dated campaigns
  under [`benchmarks/`](benchmarks/), with design notes in [`docs/`](docs/).
- *How do I rebuild, promote or roll it back?* [`production/README.md`](production/README.md).

## Status (2026-11-23)

| | |
| --- | --- |
| Text model | `qwen3.8-pi`: the harness-hub rank-32 Pi LoRA (revision `3cad999`) over the HyperQwen Qwen3.8-27B W4A16-AutoRound base, preloaded and resident (`ttl: 0`). See GitOps `PI-ADAPTER.md` |
| Runtime | HyperQwen vLLM behind llama-swap, started by `/app/single-user/start_qwen.sh` (`/app/venv`) |
| Profile | `SPEC=mtp` with 4 drafts, `MAX_LEN=172032` (168K incl. output), `PTH_FA=1`, `INT8_ACT=int8` on MLP, `MAX_SEQS=4`, `GPU_UTIL=0.95`, text only, prefix caching on 16-token match units with shared-prefix checkpoints, FP32 Mamba state, thinking off |
| On-demand model | `inkling-2-mini-unbound`: Inkling 2 Mini plus the 803 KB Unbound `edit.pt`, on the pinned [Hiutaky vLLM Ampere port](https://github.com/Hiutaky/vllm/tree/6e9d1cce3eadb9844d491de9b65dba24b9fec566), 256K context, FP8 KV, DSpark speculation, SM86 kernel overlay `inkling-sm86-opt/`, up to 8 concurrent requests. Selecting it unloads Pi |
| Image | `127.0.0.1:5000/home-llm-strata:sparsekv-v1` (layer chain in [production/README.md](production/README.md#2-images)); init container `orcasaq2-vllm` sets the GPU profile (300 W, stock clocks since 2026-11-12) |
| Weights | Text, Pi adapter, Qwen-Image and OCR from `/srv/llama-nvme/production-models` (root NVMe, staged by `prepare-ssd-models.sh`); Inkling from the second NVMe. Originals stay in the model library (GitOps `docs/model-storage.md`) |
| Other backends | `qwen-image-2.1{,-turbo}`, `glm-ocr-hires`, `paddleocr-vl{,-official}`, `finance-ocr-glm`, for their applications. Loading one evicts the resident worker |

Clients request `qwen3.8-pi` or `inkling-2-mini-unbound`. Long requests use the
authenticated Tailnet route; the public Cloudflare route times out at 125 s.

## Production timeline

Dates come from GitOps commits. "Evidence" points at the campaign that justified
each change.

| From | Text model and engine | Evidence |
| --- | --- | --- |
| 2026-08-15 | Qwen3.8-27B on patched llama.cpp | [docs/journey-2026-08-15-to-17.md](docs/journey-2026-08-15-to-17.md) |
| 2026-08-20 | Qwen3.8-27B on the syv-ai vLLM stack (v4 to v10) | [docs/vllm-companion.md](docs/vllm-companion.md), [benchmarks/qwen-vllm-hillclimb-2026-08-28](benchmarks/qwen-vllm-hillclimb-2026-08-28/) |
| 2026-09-02 | Back to llama.cpp `cuda-swap-v14` (master `0f3a71be1` + patches) | [benchmarks/engine-trial-2026-09-02](benchmarks/engine-trial-2026-09-02/) |
| 2026-09-15 | Swift Qwen3.8 Q4_K_S + DFlash2 on `cuda-swap-v25` | [archive/swift-qualification-2026-09-15.md](archive/swift-qualification-2026-09-15.md) |
| 2026-09-25 | OrcaSAQ2 EXL3 3.21 bpw (exllamav3, MTP), W4A16 as rollback | [benchmarks/orcasaq-vs-w4-2026-09-25](benchmarks/orcasaq-vs-w4-2026-09-25/) |
| 2026-10-01 | Reset: ThinkingCap Q4_K_S on upstream llama.cpp | [benchmarks/local-llm-reset-2026-09-30](benchmarks/local-llm-reset-2026-09-30/), [benchmarks/orca-thinkingcap-2026-10-07](benchmarks/orca-thinkingcap-2026-10-07/) |
| 2026-10-14 | Engine bake-off: HyperQwen vLLM wins throughput, sglang-exl3 wins overall | [benchmarks/qwen38-bakeoff-2026-10-14](benchmarks/qwen38-bakeoff-2026-10-14/) |
| 2026-10-16 | Flash Next GYRO-M on Strata (200K) added | GitOps `optimization/` |
| 2026-10-21 | HyperQwen qualified (vision, long context) and preloaded beside ThinkingCap | GitOps `hyperqwen/qualification.json`, `optimization/` |
| 2026-10-31 | Swift 1.5 Flash Next Heretic on Strata 0.2.8; other LLMs retired | GitOps `README.md` |
| 2026-11-01 | Strata/Swift decommissioned; HyperQwen is the sole text model | GitOps `README.md` (memdoor arc196_c 40/40 vs 35/40; Strata host-RAM OOM) |
| 2026-11-03 | `PTH_FA=1` promoted (E1: decode +22%, quality 42/42) | GitOps `optimization/research/coding-inference-2026-11-03/` |
| 2026-11-04 | HyperQwen weights served from root NVMe (E3: cold boot 99 s vs 208 s) | same |
| 2026-11-10 | HyperQwen served from the INT8 `lm_head` build | GitOps `2360255c` |
| 2026-11-11 | SparseKV/MTP7 rollout failed qualification and stalled; November 10 production restored | GitOps `benchmarks/sparsekv-production-2026-11-11/`, `image/sparsekv/`, PR #227 |
| 2026-11-12 | SparseKV retried with bounded observations; GPU profile moved to 300 W at stock clocks | GitOps PRs #233, #234 |
| 2026-11-14 | Cyber fine-tune (QLoRA on abliterated Qwen3.8-27B) served on demand as IQ4_XS GGUF, 128K, bounded reasoning | GitOps llama `README.md` ("Cyber fine-tune") |
| 2026-11-19 | Production model files served from root NVMe; Orca Q8 GGUF runtime with adaptive DFlash at 160K | GitOps `benchmarks/model-storage-2026-11-19-nvme/`, `image/orca-gguf/`, PRs #281, #282 |
| 2026-11-20 | `qwen3.8-pi` (Pi LoRA, MTP4, four admission slots) becomes the preloaded text model | GitOps `PI-ADAPTER.md`, `benchmarks/qwen-pi-cache-2026-11-20/`, PRs #292-#294, #297 |
| 2026-11-20 | Inkling 2 Mini added on demand (NInfer) | GitOps PRs #302, #303, #305, #308 |
| 2026-11-21 | Inkling 2 Mini moved to the pinned vLLM Ampere port; NInfer bundle kept for rollback | GitOps PR #322 |
| 2026-11-22 | Inkling SM86 profile (256K, 8-way concurrency, v3 kernels); `inkling-2-mini-unbound` is the sole Inkling route; Pi window qualified at 168K | GitOps `inkling-sm86-opt/`, PRs #332, #334 |

The Swift 1.5 Flash Heretic weights remain in
`/data/models/qwen3.8/local/swift-1.5-flash-heretic-gyro-m`, and the Strata
runtime is still in the image, so restoring Strata is a config change.

## Repository map

**Production.** [`production/`](production/) reproduces the running service:
the GitOps mirror, `LOCK.json` (file, image and weight digests), `STATUS.md`,
and [`recipes/`](production/recipes/) for layers built only from here
(`gpu-profile`, `hyperqwen`, `qwen-optimized`). `tools/production/` syncs and
verifies it; CI runs `sync.py verify` on every push.

**Harness.** [`benchmarks/paired-harness/`](benchmarks/paired-harness/) is the
paired quality and performance harness that GitOps qualification scripts call.
Do not move it.

**Evidence.** [`benchmarks/`](benchmarks/) holds one dated directory per
campaign: runner, frozen inputs, raw results and decision.
[`benchmarks/README.md`](benchmarks/README.md) indexes them by outcome.

**Design notes.** [`docs/`](docs/) holds performance
([`PERFORMANCE.md`](docs/PERFORMANCE.md)), the August journey and journal,
quant selection, DFlash2, MMQ small-batch analysis, truncated draft vocabulary
design, the SGLang claim check, and thermals/overclock
([`thermals-and-oc.md`](docs/thermals-and-oc.md), measured before the current
`gpu-profile` power settings).

**Tools and data.** [`tools/`](tools/) holds the draft-vocabulary pipeline
(`build_draft_vocab.py`, `truncate_drafter.py`, `validate_drafter.py`),
GGUF I/O, payload and session benchmarks, and `tools/bench`. [`data/`](data/)
holds the 32k/40k/48k draft-vocab keep-sets, token frequencies and coverage.

**Archive.** [`archive/`](archive/) holds superseded material, kept for
provenance: the llama.cpp v15 and syv-ai vLLM build trees, the Spark
qualification, and older patch sets. See [`archive/README.md`](archive/README.md).

## Reading old documents

Campaign and archive documents describe the system at their date. Many name
things that have since been removed:

- llama.cpp and vLLM image tags pruned from the local registry on 2026-11-04.
  Only `llama:cuda-swap-v23`/`v26` and `qwen38-27b-3090:v11` remain.
- `k8s/workloads/apps/llama/configmap.yaml` (removed 2026-11-04; the live file
  is `swap-configmap.yaml`) and `k8s/MIGRATION_LOG.md` (removed 2026-10-21, GitOps `a8603ab6`).
- The lab-root `patches/` directory (removed 2026-09-02).
- GitOps build scripts and recipes removed on 2026-11-08 (GitOps `f13dde9d`):
  `scripts/build-llama-image.sh`, `scripts/build-qwen38-vllm-{base,image}.sh`,
  `scripts/build-orcasaq-image.sh`, `k8s/workloads/apps/llama/hyperqwen/Dockerfile`
  and `optimization/build.sh`. Copies of the ones still needed are in
  `production/recipes/`, and the rest in `archive/*/scripts/`.
- Lab paths moved on 2026-11-23: `Dockerfile` and `patches-v15/` to
  `archive/llama-cpp-v15/`, `vllm/syv-ai/` and `vllm/image-build/` to
  `archive/vllm-qwen38/`, `vllm/image-orcasaq/` to
  `production/recipes/gpu-profile/`, `engines/spark/` to `archive/engines/spark/`,
  and `PROMOTE.md` to `production/README.md`.
- Models served for a while and since retired from `swap-configmap.yaml`:
  `qwen3.8-27b-hyperqwen` (replaced by `qwen3.8-pi` on 2026-11-20), the Orca Q8
  GGUF route, `qwen3.8-27b-cyber`, and the NInfer Inkling route.
- Model directories deleted on 2026-11-04: `trials/`, `k2-horizon`,
  `flash-next-coder`, `agnes`, `millie`, `gemma4`, `spark-x2.5`,
  `kvarn-port`, `muse-glimmer`, `qwen3.6/{llmfan46,local}`, `oui-1/local`, and
  `qwen3.8/local/{vllm-trial,legacy,legacy-pre-v3}`. Also deleted then: the bulky
  parts of `/data/buttercup_6tb/k3s/vllm-trial` (its scripts and reports remain).

Use `git log` in either repo to recover a removed file. Numbers from different
campaigns are only comparable when model bytes, runtime, sampling, context
depth and GPU profile all match.

## Credits

The llama.cpp patches follow llama.cpp's MIT license. Vendored trees
(`archive/vllm-qwen38/syv-ai/`) keep their upstream licenses. Prior art this work builds on:
[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)
(the truncated draft vocabulary and the vLLM stack),
[HyperQwen](https://github.com/syv-ai/HyperQwen),
[Strata](https://github.com/Niko1221/Strata),
[llama.cpp](https://github.com/ggml-org/llama.cpp),
[llama-swap](https://github.com/mostlygeek/llama-swap),
[EAGLE](https://github.com/SafeAILab/EAGLE) (the `d2t` mechanism), and
[club-3090](https://github.com/noonghunna/club-3090). The archived README has
the full credits list.
