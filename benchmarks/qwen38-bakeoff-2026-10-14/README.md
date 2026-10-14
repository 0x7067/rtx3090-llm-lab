# Qwen3.8 engine bake-off (2026-10-14): setup

Status: **set up, not run.** No GPU process was started during setup (a start-up fit check was
blocked by the session's permission policy), so nothing below has touched the GPU yet. Research context:
`docs/research/qwen38-27b-engines-quants-2026-10-14.md`,
`docs/research/qwen38-family-flash-next-2026-10-14.md`,
`docs/research/qwen38-3090-community-reports-2026-10-14.md`.

The question: on this RTX 3090, which engine serves base Qwen3.8-27B best, and how does
Qwen3.8-Flash-Next compare? All 27B arms use **base** Qwen3.8-27B weights (no finetunes)
so the comparison is about engine and quant. Past local results (2026-09-30 reset suite,
llamAmpere qualification) are context only; this campaign re-measures everything on
current builds.

## Arms

| Arm | Engine (pinned) | Weights (pinned) | Key settings |
|---|---|---|---|
| `llamacpp_q4ks` | llama.cpp b11580 `4413a1924`, `ghcr.io/ggml-org/llama.cpp@sha256:c4348afb…` (newest b-build with a CUDA server image on 2026-10-14) | bartowski Q4_K_S @`87410e7b` (`668942fa…`, has embedded nextn/MTP layer) + mmproj f16 (`cb0a71fe…`) | ctx 135,168, q8_0 K/V, MTP n=2, `--fit off`, 1 slot |
| `llamacpp_gsq` | same image | ISTA GSQ-RCO IQ3_S-mtp @`d562806d` (`58fd8267…`) + mmproj BF16 (`13cb7beb…`) | ctx 200,704, q8_0 K/V, MTP n=2, `--fit off`, 1 slot |
| `vllm_hyperqwen` | HyperQwen `95ee3b64` (patched vLLM 0.31.0), `ghcr.io/syv-ai/hyperqwen@sha256:e99df7e6…` | HyperQwen `prepare` output from dbirks W4A16 AutoRound @`1f05c441` (int8 lm_head/embed/MTP, draft vocab, fast variant) | publisher `single` profile, `SPEC=mtp CTX=long`, `PREFIX_CACHE=1`; resolved argv (`hyperqwen-argv-dryrun.txt`): serves the `-fast` variant, 150k ctx, 8 seqs, fp8 KV, MTP k=3, **text only** (`--language-model-only`), `qwen3` / `qwen3_coder` parsers |
| `sglang_exl3` | sglang-exl3 v0.7.0-ampere, `ghcr.io/0xsero/sglang-exl3@sha256:c8922bd7…` | turboderp EXL3 3.00 bpw @`6fe61ad6` (has `mtp.*` tensors) | publisher launch profile `sglang-qwen3.8-27b-exl3-3bpw-mtp-vision-208k` verbatim except port: ctx 212,992, fp8 KV, NEXTN 3/1/4 + hot-32k token map, vision on, 1 running request |
| `strata_flashnext_iq3xxs` | Strata v0.1.44.1 `296b59d2`, built locally (`CUDA_ARCHITECTURES=86`) as `local/strata-bakeoff:v0.1.44.1` | ISTA Flash-Next GSQ-RCO IQ3_XXS @`ed59f920` (`219ea929…` + n-gram shard `316b46f3…`) | ctx 150,000, int8 KV, `LOW_RAM=on` (44 GiB container cap), text only; weights + Strata data on the SSD |

Exact artifact IDs, hashes and launch flags: `arms.json` (frozen into `plan.json`),
`images.lock`, `weights.sha256`, and `run-bakeoff.sh`.

Not tested here, by design: llamAmpere (a fork; the question was upstream engines),
ExLlamaV3/TabbyAPI (covered indirectly by the EXL3 weights on SGLang), DFlash2 drafters.

## What each arm is measured on

1. **Quality**: the frozen 58-case suite from `../local-llm-reset-2026-09-30/plan-16384.json`
   (20 coding, 16 tool replay, 6 retrieval up to ~115K prompt tokens, 6 extraction, 6 other,
   4 legacy perf probes), unchanged: temperature 1.0, top-p 0.95, top-k 20,
   `reasoning_effort` low for coding / medium otherwise, 16,384-token output cap.
   Only the Python grader image was re-pinned (the old one is gone):
   `python:3.12-slim` → `sha256:df18428d…`. Plan SHA-256 is printed by `build-plan.py`.
2. **Speed** (`speed-probe.py`, after the quality run, same container): single-stream
   prose and code at short context; single-stream at ~63K and ~119K prompt tokens
   (filler calibrated with the Qwen3.8 tokenizer); prefill rate = prompt tokens / TTFT
   on nonce-prefixed (uncached) prompts; 4 concurrent callers, aggregate tok/s.
   512 output tokens, `reasoning_effort` low, same sampling. Arms with one slot
   (llama.cpp, SGLang profile, Strata) serialise the 4 callers; that is the measured
   result, not an error.
3. **Resources**: `monitor.sh` (host MemAvailable, PSI, GPU MiB, container memory;
   aborts the arm below 12 GiB available or PSI > 10) and a GPU telemetry loop
   (temp, power, VRAM, util, clocks, thermal-trip flag) every 5 s.

## Verified vs unverified (as of setup)

Verified without a GPU:
- All images pulled/built and pinned (see `images.lock`); llama.cpp image reports build 11580.
- Weight hashes match the publishers' LFS SHA-256 (`weights.sha256`).
- bartowski Q4_K_S and ISTA IQ3_S-mtp GGUFs contain the `blk.64.nextn.*` MTP tensors;
  turboderp EXL3 3.00 bpw contains `mtp.*` tensors.
- `plan.json` freezes (58 jobs × 5 arms); harness unit tests pass; the Python grader
  sandbox runs on the new image (positive control 1.0, wrong answer 0.0, network blocked)
  — `grader-dryrun.txt`.
- `speed-probe.py` parses streaming + usage correctly against a local mock server;
  filler sizes checked with the Qwen3.8 tokenizer (63,189 / 119,158 tokens).
- `run-bakeoff.sh` passes `bash -n`.
- HyperQwen `prepare` completed on CPU (`hyperqwen-prepare.log`); its launcher dry run (`PRINT_ARGV=1`)
  resolves the argv in `hyperqwen-argv-dryrun.txt` and served name `qwen3.8-27b`.
- Every SGLang flag in the publisher profile is accepted by the image's `launch_server --help`;
  the hot-32k token map exists in the image.
- Strata v0.1.44.1 pins the same ISTA revision (`ed59f920`) and accepts the published IQ3_XXS shard names via `GGUF_DIR`.

Unverified until the run (the GPU was not used during setup):
- That each arm starts and fits in 24 GB at its context (notably `llamacpp_q4ks` with
  mmproj at 135K, and `llamacpp_gsq` at 200K). `--fit off` means an over-allocation
  fails loudly instead of silently shrinking context.
- That HyperQwen's first boot finishes its torch.compile/FlashInfer JIT within the
  60-minute ready window, and that the `prepare` output is complete (see `hyperqwen-prepare.log`).
- That each engine honours `top_k`, `min_p` and `reasoning_effort` (the harness records
  failures for rejected parameters but cannot detect silently ignored ones; check server logs).
- Strata's first start: it hashes the shards, downloads its MTP layer from Hugging Face and builds
  its pack in `/data/bakeoff-ssd/strata-data` (counts against the ready window, not the measurements).
- `weights-hyperqwen.sha256` lacks one file (`-fast/model-00006-of-00007.safetensors`, written root-only 0600 by prepare).

## How to run

Prerequisites (yours):
1. **Free the GPU**: scale the production `llama` Deployment to 0 (and pause Flux for it)
   so `nvidia-smi --query-compute-apps=pid --format=csv,noheader` prints nothing.
   This script never touches k8s.
2. Keep ≥ 45 GiB host RAM free for the Strata arm (it is capped at 44 GiB and the
   monitor aborts below 12 GiB available).
3. Be at the terminal for the `sudo` prompt (matched GPU profile), or set `GPU_PROFILE=normal`.

```bash
cd /data/docker-services/rtx3090-llm-lab/benchmarks/qwen38-bakeoff-2026-10-14
./run-bakeoff.sh                         # all five arms, ~6-10 h
./run-bakeoff.sh sglang_exl3 llamacpp_q4ks   # or a subset
```

Results land in `results/<arm>/` (STATUS, quality-run/, summary.{json,md}, speed.json,
telemetry, server.log); `aggregate.py` writes `results/aggregate.{json,md}` at the end.
An arm that fails to start or run is recorded in its STATUS and the next arm runs.
A thermal trip stops the campaign. On exit the GPU profile is restored
(`sudo ../local-llm-reset-2026-09-30/gpu-cool-profile.sh restore`).

Afterwards: restore production llama (scale back up, resume Flux), and write `REPORT.md`
from `results/aggregate.md` plus per-arm failures and caveats.

## Caveats baked into the design

- One seed, one run per arm, authored synthetic suite: differences of a case or two are
  not evidence of a quality difference.
- Speeds are single observations per probe; MTP acceptance depends on content.
- Different context ceilings are deliberate (each arm's practical maximum); every arm
  must still admit the suite's ~131K deepest request.
- Model weights for the 27B arms load from the HDD-backed `/data/models` (slower startup,
  no effect on serving speed). Flash-Next lives on the SSD, which is also the etcd disk.
