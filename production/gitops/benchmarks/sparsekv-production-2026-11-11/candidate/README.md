The candidate overlays a 13 MiB archive onto the cached production image in a disposable containerd snapshot. The upstream HyperQwen launcher and SparseKV installer are unmodified beyond the published base patch. The target weights, chat template, vision settings, context limit, request slots and process limits match production. The candidate uses the published SparseKV tier with MTP-7 and FA2 prefill, while the production arm uses DFlash2-7 and its existing PTH CPU prefix cache.

- [SparseKV commit](https://github.com/bradj00/sparsekv-hyperqwen/tree/29595e8712520ef017c51bcccb9ab6dbc3c68a01) and [license](https://github.com/bradj00/sparsekv-hyperqwen/blob/29595e8712520ef017c51bcccb9ab6dbc3c68a01/LICENSE)
- [HyperQwen base commit](https://github.com/syv-ai/HyperQwen/tree/ce9615c4f2309be727407918a6c979bd67968069)
- [PTH library source pin](https://github.com/noon-at-cgn/pth-flash-attn/tree/b0d4f7ee228448ecc3d85a2015db208ed8d21954)

Run `../prepare-candidate-source.sh` on a fresh checkout. It fetches the two pinned source commits, checks the model basis hash and records its private build directory in `../.scratch`. Then run `../build-candidate.sh` to build against the locally cached production image without GPU or network. Run `../launch-arm.sh baseline --check` and `../launch-arm.sh candidate --check` to validate both launchers without CUDA. Both passed here. The CUDA image has the stable libtorch extension and emits the same benign `vllm._C` import message on the CPU-only platform with or without the overlay. PTH gate and CUDA cubin checks passed. GPU startup and performance remain unverified.

For the benchmark window, the foreground launch modes are `../launch-arm.sh baseline` on loopback port 5890 and `../launch-arm.sh candidate` on loopback port 18091. Both use a private cache, 12 CPUs, a 36 GiB memory limit, 24 GiB shared memory, and UID 1000. The model name is `qwen3.8-27b`. The benchmark harness handles GPU ownership, starts the arms serially, and restores production.

The build directory is temporary. Keep it until the benchmark is complete. Retain the recipes and evidence here after removing it and the `.scratch` pointer.
