# Build, promote, roll back

Everything needed to build a serving image for the RTX 3090 lives in this repo.
The GitOps repo (`docker-services`, where this repo is the `rtx3090-llm-lab`
submodule) holds only the Kubernetes manifests, and promotion is a tag bump
there. Images are imported into k3s containerd directly (no registry), so a tag
must be bumped on every rebuild: `imagePullPolicy: Never` will not re-import a
same-tag rebuild.

## llama.cpp (production since 2026-09-02)

Recipe: [`Dockerfile`](Dockerfile) at the repo root, patches from
[`patches-v14/`](patches-v14/). Bump `LLAMA_CPP_REF` and re-run the rebase
procedure in [`patches-v14/REBASE-2026-09-02.md`](patches-v14/REBASE-2026-09-02.md)
when moving the base.

```bash
# from the docker-services checkout
IMAGE_NAME=llama:cuda-swap-v15 scripts/build-llama-image.sh   # builds from rtx3090-llm-lab/, imports into k3s
```

Validate before serving (both must print `Backend CUDA0: OK`):

```bash
docker run --rm --gpus all -e GGML_CUDA_FATTN_MMA_Q=1 --entrypoint /usr/local/bin/test-backend-ops llama:cuda-swap-v15 -b CUDA0 -o FLASH_ATTN_EXT
docker run --rm --gpus all --entrypoint /usr/local/bin/test-backend-ops llama:cuda-swap-v15 -b CUDA0 -o MUL_MAT
```

Serving profile: the `qwen3.8-27b` stanza in the GitOps ConfigMap
(`k8s/workloads/apps/llama/configmap.yaml`), mirrored here as
[`config/llama-swap-qwen38.yaml`](config/llama-swap-qwen38.yaml). Coupled
settings (env caps + draft length) are documented inline there.

## vLLM (production 2026-08-20 to 2026-09-02, kept as rollback)

Source: [`vllm/syv-ai/`](vllm/syv-ai/), a squashed subtree of
`syv-ai/qwen38-27b-rtx3090` at our `k8s-deploy-v10` overlay (syv-ai `453104e`).
Pull upstream with
`git subtree pull --prefix=vllm/syv-ai https://github.com/syv-ai/qwen38-27b-rtx3090 main --squash`.
Build contexts: [`vllm/image-build/base/`](vllm/image-build/base/) (locked
197-package environment, `Dockerfile.locked`) and
[`vllm/image-build/overlay/`](vllm/image-build/overlay/) (vLLM patches applied on
a built base).

```bash
scripts/build-qwen38-vllm-base.sh qwen38-27b-3090:v12-base --check   # dry run of the context
scripts/build-qwen38-vllm-base.sh qwen38-27b-3090:v12-base --publish
scripts/build-qwen38-vllm-image.sh qwen38-27b-3090:v12 --import       # overlay patches on the base
```

The last deployed vLLM manifest is
[`vllm/image-v10/deployment-k8s-last.yaml`](vllm/image-v10/deployment-k8s-last.yaml).

## Benchmark before promoting

[`benchmarks/engine-trial-2026-09-02/bench/`](benchmarks/engine-trial-2026-09-02/bench/)
is engine-agnostic (OpenAI-compatible) and covers decode, sustained generation,
cold prefill, agentic edit sessions at 0/20k/50k preamble, 4-way concurrency,
and a 4-task quality battery:

```bash
python3 bench.py selftest
./run-arm.sh <tag> http://127.0.0.1:<port>/v1      # appends to results.jsonl
python3 bench.py report results.jsonl
```

Pin `--reasoning medium` across arms (the harness does), interleave a repeat of
the control arm, and read the fresh-server warm-up run as the honest
single-request decode number for any `ngram-mod` arm: repeated identical
prompts replay from the n-gram table and inflate to 300-400 tok/s.
`run-window.sh` in that directory is the production-window runbook (suspend
Flux, scale to zero, arms, restore).

## Promote

1. Build and import the image (above), bump the tag in
   `k8s/workloads/apps/llama/deployment.yaml` (and `configmap.yaml` if the
   profile changes), one line in `k8s/MIGRATION_LOG.md`.
2. `git push` then `flux reconcile kustomization apps --with-source`.
3. Verify: `kubectl -n apps rollout status deploy/llama`, one request through
   `http://<home-server-overlay-ip>/v1` with the bearer token, then a short `bench.py decode`
   and `bench.py session` against the service.
4. Clients (Pi / Prime on home-server, mac-studio, the Ravn
   MacBook) read `~/.pi/agent/models.json` and `~/.prime/agent/models.json`;
   keep `contextWindow` equal to the served context.

## Roll back

`git revert <promotion commit>` in the GitOps repo and reconcile. Previous
images stay in containerd (`sudo k3s ctr images ls | grep -E 'llama:|qwen38'`),
so a rollback needs no rebuild. If the previous engine was vLLM, set
`llama-cache-canary` back to `replicas: 1`.
