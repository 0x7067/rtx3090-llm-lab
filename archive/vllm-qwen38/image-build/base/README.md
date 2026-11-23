# Locked vLLM base image

`requirements.lock` contains the complete 197-package Python environment from
the qualified v7 image, including Torch, Triton, FlashInfer, xgrammar, and every
`nvidia-*` and `cuda-*` wheel. `uv` generated hashes for every artifact.

`Dockerfile.locked` also pins the CUDA base-image digest, pip, Python 3.12,
CUDA nvcc/runtime development packages, and the build tools used by v7.

Check the overlaid full-build context without building:

```bash
scripts/build-qwen38-vllm-base.sh ignored:check --check
```

Build and publish a new full base through the local registry:

```bash
scripts/build-qwen38-vllm-base.sh qwen38-27b-3090:v8 --publish
```

Refresh the lock only after qualifying a replacement image:

```bash
scripts/lock-qwen38-vllm-deps.sh qwen38-27b-3090:v8
git diff -- k8s/workloads/apps/llama/vllm-base/
```

The lock refresh treats the named image as the authority. It does not resolve
newer transitive packages unless they are already installed in that image.

## v10 (2026-09-01)

Built with this Dockerfile from the df2-repo checkout on branch
`local/k8s-deploy-v10` = local commits rebased onto syv-ai main `453104e`.
`requirements.lock` is unchanged (vLLM 0.27.1). Two apt pins moved because the
Ubuntu archive dropped the old revisions: `python3.12*` `3.12.3-1ubuntu0.15 ->
0.16` and `curl` `8.5.0-2ubuntu10.12 -> 10.13`. If the build fails in the
apt layer again, `apt-cache policy <pkg>` inside the pinned base image shows the
current candidate; bump the pin, do not drop it.

The `vllm-patches/` overlay Dockerfile (v8 -> v9) is not used for v10: both of
its patches are in the branch's `patches/` directory and applied by this
Dockerfile's patch loop. Upstream's `xgrammar-spec-terminated.patch` was
dropped from the branch in favour of our `xgrammar-terminated-batch.patch`
(a superset that conflicts with it textually).
