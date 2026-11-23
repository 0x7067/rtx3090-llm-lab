# Preparation evidence — 2026-11-11

The published image is `127.0.0.1:5000/home-llm-strata:sparsekv-v1@sha256:ecb915cc18b92f1c81dcc97d22576bac3eacba436afc00fb7afeb6d1ba604679`.

- `build.log`, `build.exit`, and `build-metadata.json`: successful offline CPU compilation, PTH gate tests, SparseKV install verification, private-runtime imports, unchanged original runtime source hashes, and publication.
- `runtime-archive.sha256`: hash of the generated 201 MiB uncompressed added runtime. The temporary archive and extracted source/runtime directories were removed after qualification.
- `runtime-check.json`, stdout/stderr, and `published-check.exit`: the published image checked as UID 1000 in a disposable container without GPU access. The launcher argv matches the benchmark except for the port; llama-swap accepts the config.
- `checks.log` and `checks.exit`: repository tests, supply-chain checks, both Kustomize renders, shell checks, Python parsing, and semantic equality of all other model entries. The final digest-pinned manifests also passed supply-chain and render checks.
- `build-cache-cleanup.exit`: successful removal of the disposable HDD builder cache. The builder container and remote builder registration were removed; no benchmark/build container remains.

The cancelled NVMe/HDD build logs and the two failed attempts are retained for diagnosis. The final build reuses k3s's already-unpacked base instead of copying/unpacking it again. Source staging uses `patch --fuzz=0` explicitly: `git apply` inside the nested untracked source directory had silently skipped the patch. The corrected compiled kernel and every SparseKV patch passed the final checks.

Production is still on the original DFlash image. No GPU execution or live service switch was performed for this preparation. Live qualification in `qualify_live.py` remains pending the approved rollout, including image input, four clients, cancellation recovery, and near-window retrieval. The earlier benchmark evidence is in PR #215.
