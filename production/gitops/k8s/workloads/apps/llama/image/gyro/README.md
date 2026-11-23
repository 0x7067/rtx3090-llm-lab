# Flash Next GYRO-M on the home RTX 3090

Selectable model: `qwen3.8-flash-next-gyro-m`, through the existing authenticated
`https://llama.denguinho.org/v1` endpoint. The default 27B model and its aliases are
retained. llama-swap keeps one backend resident at a time.

The profile follows the supplied screenshot: GYRO-M, 200,000 total context tokens,
K8/V4 cache (`k8v4`), thinking enabled by default, temperature 1.0, top-p 0.95,
top-k 20, min-p 0.05. Explicit request settings still take precedence. It has a
Strata MTP draft, automatic GPU expert cache, and a 20 GiB resident CPU expert
ceiling. Text and tools are supported; vision is not configured.
Strata binds only to pod loopback; its hostname allowlist includes
`llama.denguinho.org` for requests through the authenticated proxy. The
2026-10-18 host calibration sets a 20 GiB resident ceiling and adapts every
decode window with up to 160 swaps at 0.97 decay. On this RTX 3090 profile,
the GPU already holds enough experts that Strata can use only 16.03 GiB of
the requested resident budget; the higher value is a ceiling, not 20 GiB of
observed allocation.

## Pinned artifacts

- Image: `127.0.0.1:5000/home-llm-gyro@sha256:61d509794b6e39de19d1b33bb49e4ecd544bf0f4417b64caeb5158c2b5cd9110`.
- [Model](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF), revision
  `aa0a0c3e24f9e08716ae806d40489abc7db6237f`, file
  `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf`, 91,965,774,272 bytes.
- Model SHA-256: `7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f`.
- [Strata rc2](https://github.com/agentionai/Strata/tree/rc2):
  `9ff02c9fa9c86a88088370925b3c9131cc5d6c26` (engine 0.1.40.3).
- [GGML fork](https://gitlab.com/agentionai-group/llama.cpp):
  `9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f`, CUDA 13, sm_86.

Weights, pack, draft and their provenance live at `/data/gyro-m`, mounted read-only
in the pod. The 1.38 GiB pack uses upstream `iq_pack.py --compat-bf16`: 363 dense
tensors are converted to BF16; expert and n-gram table bytes are unchanged. The
draft reuses the existing Strata Q2_0 MTP runtime; its vocabulary matches this
build byte-for-byte. `/data/gyro-m/provenance.json` records the draft hashes.
This format needs the pinned custom runtime; stock llama.cpp cannot load it.

## Validation on 2026-10-16

Full-file SHA-256 and native tensor compatibility passed (zero failures).
Staging passed streaming arithmetic, reasoning, the exact Python code contract,
parsed tool call and tool-result follow-up. Retrieval of three keys from a
189,998-token prompt passed in 121.415 seconds; the continuation reused 189,993
tokens and passed in 2.338 seconds. These are functional checks, not a general
quality ranking or validation of the screenshot's SOTA claims.

Staging used a 40 GiB container memory limit, no swap, eight CPUs, and the host
thermal guard. Observed peak VRAM was 23,839 MiB and minimum available host RAM
was 24.54 GiB. Raw synthetic test responses and telemetry remain local at
`/tmp/gyro-release/`; the compact report is `qualification.json` here.

Production passed authenticated streaming, reasoning, the code contract, tools
and follow-up, public inference, client catalog discovery, and switching to the
retained `qwen3.8-27b` alias and back. Both routes reject unauthenticated requests
with 401. The initial public inference 403 was resolved by the explicit hostname
allowlist. GYRO-M cold loads took 87–109 seconds; the default EXL3 cold load took
157 seconds. GYRO-M is left warm after verification.

## Host calibration on 2026-10-18

Strata's pinned calibrator retained only `--adapt-every 1 --adapt-swaps 160
--adapt-decay 0.97`. Its 512-token calibration median improved from 36.3 to
60.1 tok/s. The automatic PCIe share (0.35), draft floor (0.5), and nine-worker
CPU pool won their confirmation checks and remain implicit/default.

A matched 4K/32K three-run benchmark measured the calibrated synchronous
profile at 69.4/63.8 tok/s median decode, versus 64.6/28.3 tok/s for the prior
profile; the old 32K arm included 4.3 and 28.3 tok/s file-tier stalls. All short
functional checks passed. The final 189,998-token retrieval passed in 121.74 s,
and its 189,993-token cached continuation passed in 2.08 s.

`--adapt-async 1` is not deployed. Its first request decoded at 64.7 tok/s, but
host memory PSI crossed the 10% safety threshold immediately afterward, so the
guard stopped the arm and production was restored. Compact evidence is in
`tuning-2026-10-18.json`; raw responses and telemetry remain local under
`/tmp/gyro-tuning-20261018.0zz3oS/`.

## Rebuild and rollback

Run `bash build.sh` from this directory or supply its full path. It verifies the
existing build-base image identity, fetches pinned sources, and adds the runtime
to the previous production image. Existing backends keep their runtimes.
Publish a new tag with `scripts/publish-image-to-k3s.sh`, update the deployment
digest, push and reconcile Flux.

To test again, run `bash stage.sh /tmp/gyro-recheck` from this directory. It
temporarily pauses model serving and restores the deployment and Flux afterward.
It expects `/data/gyro-m/config/gyro-m.json` to match the ConfigMap profile.

Rollback: `git revert cb4b7ff0 c1a18273`, push, and run
`flux reconcile kustomization apps --with-source`. The previous image remains
available and model files are retained.
