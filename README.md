# RTX 3090 LLM Lab

Evidence, build sources and tooling for serving local LLMs on one 24 GB RTX 3090
(Ampere, sm_86) in the home k3s cluster. This repo is the `rtx3090-llm-lab`
submodule of the GitOps repo `docker-services`, which holds the manifests Flux
applies.

**Which repo answers what:**

- *What is running, and with which flags?* GitOps
  `k8s/workloads/apps/llama/`: `README.md`, `deployment.yaml` and
  `swap-configmap.yaml`. That is the source of truth for live state.
- *Why was it chosen, and how was it measured?* This repo: dated campaigns
  under [`benchmarks/`](benchmarks/), with design notes in [`docs/`](docs/).
- *How is an image built and promoted?* [`PROMOTE.md`](PROMOTE.md).

## Status (2026-11-04)

| | |
| --- | --- |
| Text model | `qwen3.8-27b-hyperqwen`: HyperQwen Qwen3.8-27B W4A16-AutoRound, preloaded and resident (`ttl: 0`) |
| Runtime | HyperQwen `2cf50a0` (pinned vLLM 0.31) behind llama-swap, started by `/app/single-user/start_qwen.sh` |
| Profile | `SPEC=mtp`, `CTX=long` (150K), `PTH_FA=1`, `INT8_ACT=int8` on MLP, `MAX_SEQS=4`, `VISION=1` with offload, prefix caching |
| Image | `127.0.0.1:5000/home-llm-strata:0.2.8-v1` (layer chain in [PROMOTE.md](PROMOTE.md)); init container `orcasaq2-vllm` sets the GPU profile |
| Weights | `/srv/llama-nvme/hyperqwen-bakeoff/models` (root NVMe copy); the original is in `/data/models/hyperqwen-bakeoff/models` on the HDD |
| Other backends | `qwen-image-2.1{,-turbo}`, `glm-ocr-{baseline,hires}`, `paddleocr-vl{,-official}`, `finance-ocr-glm`. Loading one evicts HyperQwen (cold reload ~100 s from NVMe) |

Clients request `qwen3.8-27b-hyperqwen`. llama-swap forwards it to vLLM as
`qwen3.8-27b` (`useModelName`).

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

The Swift 1.5 Flash Heretic weights remain in
`/data/models/qwen3.8/local/swift-1.5-flash-heretic-gyro-m`, and the Strata
runtime is still in the image, so restoring Strata is a config change.

## Repository map

**Build inputs the GitOps repo reads. Do not move these.**

- [`Dockerfile`](Dockerfile) + [`patches-v15/`](patches-v15/): llama.cpp
  `0f3a71be1` with patches 0002-0011 (`scripts/build-llama-image.sh`). Not
  served today.
- [`vllm/syv-ai/`](vllm/syv-ai/) (squashed syv-ai stack) and
  [`vllm/image-build/`](vllm/image-build/) (`base`, `overlay`): the vLLM
  `qwen38-27b-3090` images (`scripts/build-qwen38-vllm-{base,image}.sh`).
- [`vllm/image-orcasaq/`](vllm/image-orcasaq/): the `orcasaq2-vllm` image,
  whose `gpu-profile` init container runs in production
  (`scripts/build-orcasaq-image.sh`).
- [`benchmarks/paired-harness/`](benchmarks/paired-harness/): the paired
  quality and performance harness, which GitOps qualification scripts call.

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

**Engines.** [`engines/spark/`](engines/spark/) holds the Spark X2.5 4B
qualification (no longer served).

**Archive.** [`archive/`](archive/) holds superseded material, kept for
provenance; see [`archive/README.md`](archive/README.md).

## Reading old documents

Campaign and archive documents describe the system at their date. Many name
things that have since been removed:

- llama.cpp and vLLM image tags pruned from the local registry on 2026-11-04.
  Only `llama:cuda-swap-v23`/`v26` and `qwen38-27b-3090:v11` remain.
- `k8s/workloads/apps/llama/configmap.yaml` (removed 2026-11-04; the live file
  is `swap-configmap.yaml`) and `k8s/MIGRATION_LOG.md` (removed 2026-10-21, GitOps `a8603ab6`).
- The lab-root `patches/` directory (removed 2026-09-02).
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
(`vllm/syv-ai/`) keep their upstream licenses. Prior art this work builds on:
[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)
(the truncated draft vocabulary and the vLLM stack),
[HyperQwen](https://github.com/syv-ai/HyperQwen),
[Strata](https://github.com/Niko1221/Strata),
[llama.cpp](https://github.com/ggml-org/llama.cpp),
[llama-swap](https://github.com/mostlygeek/llama-swap),
[EAGLE](https://github.com/SafeAILab/EAGLE) (the `d2t` mechanism), and
[club-3090](https://github.com/noonghunna/club-3090). The archived README has
the full credits list.
