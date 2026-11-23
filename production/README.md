# Production: reproduce the running service

This directory holds everything needed to rebuild and relaunch the llama
service that is running on the RTX 3090 now, pinned to one GitOps commit.

| Path | What it is | Edited by |
| --- | --- | --- |
| [`STATUS.md`](STATUS.md) | Models, containers, images and host inputs, rendered from the files below | generated |
| [`LOCK.json`](LOCK.json) | GitOps commit, every digest-pinned image reference, SHA-256 of every file here, and a digest of each read-only host input (weights, runtime bundles) | generated |
| [`gitops/`](gitops/) | Byte-for-byte copy of the GitOps inputs: `k8s/workloads/apps/llama/` (manifests, launchers, kernel overlays, image recipes, staging scripts), `benchmarks/`, `docs/model-storage.md` and two `scripts/` | generated |
| [`recipes/`](recipes/) | Build recipes that exist only here: the `gpu-profile` image, and layers whose GitOps recipes were deleted in GitOps `f13dde9d` | by hand |

Do not edit `gitops/`, `LOCK.json` or `STATUS.md`. Change GitOps instead; the
sync copies it here.

## How it stays current

`tools/production/sync.py` exports the files listed in its `SOURCES` from a
GitOps commit, drops `k8s/workloads/apps/llama/optimization/` (research logs,
over 100 MB) and every `*.enc.yaml`, then regenerates `LOCK.json` and
`STATUS.md`. With `--host-scan` it hashes each read-only `hostPath` volume of
the pod (cached by inode, size and mtime, so later runs only hash what changed).

- **After each promotion**, run
  [`tools/production/publish.sh`](../tools/production/publish.sh) on
  `denguinho-server` from this checkout, at `origin/main` with a clean tree. It
  syncs from GitOps `main` (bare clone under `~/.cache/rtx3090-lab-sync/`) with
  `--host-scan`, runs `sync.py verify` and gitleaks (`.gitleaks.toml`), then
  commits `production/` and pushes it to this repo's `main` if anything changed.
  Nothing runs it on a schedule, so `production/` is as current as the last run;
  the GitOps commit is in `STATUS.md`.
- **On every push and pull request**, CI (`.github/workflows/production.yml`)
  runs `sync.py verify`: each file matches its hash in `LOCK.json`, the image
  list matches the files, and `STATUS.md` matches its rendering.
- **To check for drift** without writing:
  `tools/production/sync.py sync --gitops /data/docker-services --ref origin/main --host-scan --check`
  exits 1 if GitOps or a host input has changed since the last publish.

## Rebuild from scratch

Paths starting with `gitops/` are relative to this directory and match the
same path in the GitOps repo.

### 1. Host inputs

`LOCK.json` → `host_inputs` lists every read-only mount with its byte count
and `tree_sha256`. After staging, compare with
`tools/production/sync.py hash-tree <path>`.

| Input | How to stage it |
| --- | --- |
| Qwen3.8-27B W4A16-AutoRound base, Qwen-Image, OCR models and runtimes | `gitops/k8s/workloads/apps/llama/prepare-ssd-models.sh` (as root) copies them from the model library on Buttercup to `/srv/llama-nvme/production-models`. The library layout is in `gitops/docs/model-storage.md`. The base's `config.json` was edited locally after download, so match it against the tree hash, not against Hugging Face |
| Pi LoRA | `harness-hub/Qwen3.8-27B-pi-adapted` revision `3cad999e2b9ec72fea8d802cc286e4502adfa082`; the file hash is in `gitops/k8s/workloads/apps/llama/PI-ADAPTER.md`. Staged by `prepare-ssd-models.sh` |
| Inkling 2 Mini weights and vLLM Ampere runtime | `gitops/k8s/workloads/apps/llama/prepare-inkling-vllm.sh`: pins the `vllm/vllm-openai` nightly image by digest, the Hiutaky fork commit and the Hugging Face revision, and verifies both |
| Inkling Unbound edit | `nokeepai/Inkling-2-Mini-Unbound` `abliteration/edit.pt`; the launcher checks its SHA-256 on load (`INKLING_UNBOUND_EDIT` in `gitops/k8s/workloads/apps/llama/swap-configmap.yaml`) |
| NInfer rollback bundle | `gitops/k8s/workloads/apps/llama/prepare-inkling-2-mini.sh` |

### 2. Images

The `llama-swap` container runs `home-llm-strata:sparsekv-v1`. Each layer is
built `FROM` the digest of the one below it:

| Layer | Adds | Recipe |
| --- | --- | --- |
| `home-llm-sglang`, `home-llm-gyro` | sglang-exl3 base, earlier llama.cpp runtimes, Strata GYRO | `gitops/k8s/workloads/apps/llama/image/gyro/` (sglang base: GitOps history at `51405e57`) |
| `home-llm-qwen-image` | Qwen-Image runtime, llama-swap | `gitops/k8s/workloads/apps/llama/qwen-image/Dockerfile` |
| `home-llm-hyperqwen` | HyperQwen `2cf50a0` at `/app` (vLLM 0.31.0) | [`recipes/hyperqwen/Dockerfile`](recipes/hyperqwen/Dockerfile) |
| `home-llm-qwen-optimized` | Strata `011c9e6`, engine 0.1.47 | [`recipes/qwen-optimized/build.sh`](recipes/qwen-optimized/build.sh) |
| `home-llm-ocr` | OCR services | `gitops/k8s/workloads/apps/llama/ocr/` (its README, "Build and deploy") |
| `home-llm-raon` | Raon llama.cpp build | `gitops/k8s/workloads/apps/llama/image/raon/` |
| `home-llm-strata:0.2.8-v1` | Strata 0.2.8, engine 0.1.54 | `gitops/k8s/workloads/apps/llama/image/strata/build.sh` |
| `home-llm-strata:normal120-dflash-short-v1` | Short DFlash launcher patch | `gitops/k8s/workloads/apps/llama/image/strata-dflash-short/` |
| `home-llm-strata:sparsekv-v1` | SparseKV runtime at `/opt/sparsekv`; `qwen3.8-pi` keeps using `/app/venv` | `gitops/k8s/workloads/apps/llama/image/sparsekv/` |

The `gpu-profile` init container runs `orcasaq2-vllm`, built from
[`recipes/gpu-profile/`](recipes/gpu-profile/) with `build.sh`.

To change a layer, rebuild it with a new tag and rebuild every layer above it
on the new digest. Publish with `gitops/scripts/publish-image-to-k3s.sh`, and
bump the tag on every rebuild, because containerd caches by tag. A model
profile change (env, context, speculation) needs no image: it lives in
`gitops/k8s/workloads/apps/llama/swap-configmap.yaml`.

### 3. Deploy

The manifests are `gitops/k8s/workloads/apps/llama/`. Flux applies them from
GitOps `main` (`flux reconcile kustomization apps --with-source`). The
ConfigMaps the pod mounts (`start_qwen_cache.py`, `start_inkling_vllm.sh`,
`inkling-sm86-opt/`, chat templates) are generated from files in the same
directory by its `kustomization.yaml`. The secrets (`*.enc.yaml`) are not
mirrored here.

### 4. Verify

- `kubectl -n apps rollout status deploy/llama`
- `/v1/models` lists the models in `STATUS.md`
- one authenticated chat request per listed model
- the qualification records named in `gitops/k8s/workloads/apps/llama/README.md`

## Promote and roll back

Promote in GitOps: change `deployment.yaml` or `swap-configmap.yaml`, record
the evidence next to the model's doc, push, reconcile. Then run
`tools/production/publish.sh` here and add a dated row to the [README timeline](../README.md#production-timeline).

Roll back with `git revert` in GitOps and reconcile. Images are pinned by
digest, and registry clean-ups keep digests referenced by GitOps manifests and
Dockerfiles, so a revert finds its image. Check with
`curl -s 127.0.0.1:5000/v2/<repo>/tags/list` before relying on an older one.

## Known gaps

- **Bottom layers cannot be rebuilt from source.** `home-llm-gyro` builds
  from `local/strata-bakeoff:v0.1.44.1`, which no longer exists, and
  `home-llm-sglang` descends from older fleet images. The running images
  survive in the local registry because GitOps references them. An offline
  copy (for example `skopeo copy` into an OCI layout on Buttercup, about 50 GB)
  would remove that dependency.
- **Some bases are pinned by tag only:** `qwen-image/Dockerfile` (`uv`,
  `busybox`) and `recipes/gpu-profile/Dockerfile` (`VLLM_TAG`). A rebuild may
  differ from the running digest recorded in `LOCK.json`.
- **Weights are not in git.** `LOCK.json` records what they must hash to, not
  where to download every file.
- The Orca GGUF overlay (`cachefix-draft-page-priority-01`) lived only on
  disk and is gone. Orca is no longer served.
