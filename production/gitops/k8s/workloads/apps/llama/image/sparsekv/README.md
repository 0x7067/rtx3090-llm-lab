# SparseKV production profile

This image packages the SparseKV/MTP7 profile measured in
[benchmark PR #215](https://github.com/Denguinho/docker-services/pull/215).
It keeps the resident model ID, weights, chat template, 200K context, four
admissions, vision support, and llama-swap authentication. The only model
entry switched to the private runtime is `qwen3.8-27b-hyperqwen`.

## Source and packaging

`stage.sh` fetches exact commits and verifies the basis checksum:

- SparseKV: `29595e8712520ef017c51bcccb9ab6dbc3c68a01`.
- HyperQwen launcher/PTH source: `ce9615c4f2309be727407918a6c979bd67968069`,
  with SparseKV's `hyperqwen-base.diff` and release Python/patch files.
- Base image: the deployed DFlash image, pinned in `Dockerfile`.
- Basis SHA256: `dee8b28281d5837ad512a560e4d5c31af245cbe42dc1dea66dc1510775b4b99b`.

The source preparation matches the benchmark. Compilation and installation
run offline using the base image's locked dependencies. No packages install
at service startup. `/opt/sparsekv/venv` owns its vLLM Python sources and PTH
package; immutable native libraries and unrelated dependencies are symlinks
to the pinned base. No hard links are used. Its vLLM entry point invokes its
own Python interpreter. The original `/app/venv` remains available to every
other model. `check_image.py` verifies imports and every original vLLM/PTH
Python file's hash after installation.

`build.sh` compiles and checks the private runtime in a disposable containerd
container using the exact base digest already unpacked for production. It
exports only `/opt/sparsekv` and hashes that archive. The Dockerfile uses
`COPY --link` to publish the added layer without re-extracting the 36.5 GiB
compressed base. No production mounts or credentials enter the build.

The script publishes a fresh tag and records the digest in
`qualification/build-metadata.json`; it refuses an existing registry tag.
Compilation is CPU-only, two CPUs / 4 GiB. The owner has a 25-minute deadline;
publication has its own 10-minute timeout. `status.sh` reports the owning unit
and build log. The disposable `sparsekv-adoption` publishing builder uses HDD
storage, a 5 GiB cap, and no restart policy, and is removed after publication.
`check_runtime.py` then checks the published image as UID 1000 without GPU
access: imports, untouched base sources, installed llama-swap validation, and
exact argv equality with `benchmarked.argv` except for the serving port.

## Profile and expected tradeoff

`../../swap-configmap.yaml` is the source of truth for settings. The measured
profile uses accuracy-first selection each decode step, INT4 sketches and
host KV, 17.8 decimal GB host KV, 32 default GDN checkpoint slots, MTP7,
FA2 prefill, and a 0.025 hot fraction. Exact follow-up prefill and serialized
staging retain upstream defaults. Existing pod memory/CPU, shared memory,
and GPU power limits remain in force.

The production profile allocates 16 decimal GB of host KV. This raises cache
capacity by about 33% from the qualified 12 GB rollout. At preparation time,
the host had about 9.8 GiB available; the additional 4 decimal GB would leave
about 6 GiB at the same load. The pod's 26Gi request and 36Gi limit remain
unchanged. This capacity increase requires live qualification after restart.

The original 17.8 GB rollout refused allocation with only 21.3 GB available.
The subsequent 12 GB rollout passed four concurrent 31.5K-token requests,
189K-token retrieval and a cached follow-up. The upstream 8 GB allocation
reserve remains in force. Cache capacity controls retained sessions; the
200K per-request limit and selection precision stay the same.

The benchmark measured 3.0x aggregate output throughput for two simultaneous
~124K-token chats, but 15% lower single-request decode throughput at 32K.
Small retrieval/tool/structured-output screens passed; that is limited
quality evidence. The comparison changed the whole serving stack, including
DFlash7 to MTP7. It does not isolate the effect of sparse attention.

## Rollout and rollback

The single GPU requires one brief llama interruption to switch images.
Prepare and review the digest-pinned PR before that interruption. Merge only
after automated reviews, including Devin Review, have landed and comments
are addressed. Flux then applies the Recreate deployment. Keep the previous
image and config available until the new backend passes live checks.

Acceptance checks cover the actual authenticated llama-swap endpoint:
model metadata; plain and streamed completion; tools and tool continuation;
JSON; image input; four concurrent requests; cancellation followed by a
successful request; and a near-window long prompt. Check backend metrics,
SparseKV startup configuration, host memory, pod restarts, and Flux readiness.
The pod's `/health` alone does not prove that the preloaded model is ready.

If startup or checks fail, revert the adoption merge through GitOps and
reconcile `apps`; verify the original image, DFlash configuration, and a real
inference. The original image is
`127.0.0.1:5000/home-llm-strata:normal120-dflash-short-v1@sha256:83077f60377d9e303d446a2f045381daab28a123e8e314c5bbe1ba53314884a2`.
No model files or persistent data migration is involved.

## Retry after the memory-pressure incident

`memory-diagnosis/REPORT.md` records the failed rollout's memory trajectory.
The SparseKV engine stayed near 20 GiB while the workerd group rose to about
10 GiB. The retry uses the upstream launcher and its existing allocation
check, with the original pod resource limits. `monitor_memory.py` records
host, process-group, and model-cgroup memory for diagnosis during qualification.
The approved rollout must qualify near-window retrieval and follow-up and
restore DFlash if the serving checks fail.

## Stock-clock retry and normal-use diagnostics

The November 12 retry reuses the pinned image and the 16 GB host-cache profile
with the production stock-clock GPU configuration. `serve_observer.py` wraps
the upstream launcher and records progress, GPU and memory telemetry, plus
bounded captures of suspected stalls. [OBSERVABILITY.md](../../OBSERVABILITY.md)
describes retention, Prometheus metrics and capture/recovery procedures.
The earlier CUDA recorder remains available for a focused investigation.
