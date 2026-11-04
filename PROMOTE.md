# Build, promote, roll back

The production image is built and promoted from the GitOps repo
(`docker-services`). This repo supplies evidence, the paired harness and the
legacy engine sources. The runbook for the retired llama.cpp v14/v15 and vLLM
v9-v12 images is [archive/PROMOTE-llama-cpp-vllm-2026-09.md](archive/PROMOTE-llama-cpp-vllm-2026-09.md).

Paths below are relative to the GitOps repo root unless they start with
`rtx3090-llm-lab/`.

## The production image

`127.0.0.1:5000/home-llm-strata:0.2.8-v1` is a chain of layers. Each layer is
pinned by digest to the one below it:

| Layer | Adds | Recipe |
| --- | --- | --- |
| `home-llm-qwen-image` | Qwen-Image runtime, llama-swap | `k8s/workloads/apps/llama/qwen-image/Dockerfile` |
| `home-llm-hyperqwen` | HyperQwen `2cf50a0` at `/app` (vLLM 0.31.0, asserted at build) | `k8s/workloads/apps/llama/hyperqwen/Dockerfile` |
| `home-llm-qwen-optimized` | Strata `011c9e6` / engine 0.1.47 | `k8s/workloads/apps/llama/optimization/build.sh` |
| `home-llm-ocr` | OCR services | `k8s/workloads/apps/llama/ocr/` (see its README, "Build and deploy") |
| `home-llm-raon` | Raon llama.cpp build | `k8s/workloads/apps/llama/image/raon/Dockerfile` |
| `home-llm-strata` | Strata 0.2.8 / engine 0.1.54 at `/opt/strata-0.2.8` | `k8s/workloads/apps/llama/image/strata/build.sh` |

The init container `orcasaq2-vllm` (`gpu-profile`) is built from
`rtx3090-llm-lab/vllm/image-orcasaq/` by `scripts/build-orcasaq-image.sh`.

To change a layer, rebuild it with a new tag, then rebuild every layer above it
on the new digest. If you only change the HyperQwen profile (env, context,
spec mode), no image rebuild is needed: that lives in `swap-configmap.yaml`.

Notes:

- `optimization/build.sh` defaults its engine archive to
  `/data/qwen-candidates-20261021/runtime/strata-engine.tar.gz`, which does not
  exist on the host. Pass the archive as the third argument; the script checks
  its pinned SHA-256.
- The base has more than 127 layers, so plain `docker build` fails ("max depth
  exceeded"). `image/strata/build.sh` uses BuildKit with `COPY --link` and pushes
  straight to the registry.

## Publish

For an image built with `docker build -t <name>:vN <ctx>`, run
`scripts/publish-image-to-k3s.sh <name>:vN` and paste the printed
`127.0.0.1:5000/<name>:vN@sha256:…` into the manifest. `image/strata/build.sh`
publishes on its own and prints the reference. Bump the tag on every rebuild,
because containerd caches by tag.

## Qualify

Run the paired harness against the candidate before promoting:
`rtx3090-llm-lab/benchmarks/paired-harness/harness.py` (see its README). Live
checks and earlier qualification records are in
`k8s/workloads/apps/llama/optimization/` and `hyperqwen/qualification.json`. To
trial a profile, add it as a separate llama-swap entry (for example
`qwen3.8-27b-hyperqwen-pthfa` on 2026-11-03), then promote it by editing the
resident entry and removing the trial entry.

## Promote

1. Edit `k8s/workloads/apps/llama/deployment.yaml` (image digest) and/or
   `swap-configmap.yaml` (profile). Record the decision and evidence in
   `k8s/workloads/apps/llama/README.md` ("Current production"), and add a
   dated row to this repo's README timeline.
2. Push to `main`, then `flux reconcile kustomization apps --with-source`.
3. Verify:
   - `kubectl -n apps rollout status deploy/llama`
   - `/v1/models` lists `qwen3.8-27b-hyperqwen`
   - one authenticated chat request
   - a short paired-harness run

The first text request after a restart waits for the HyperQwen preload (about
100 s from the NVMe weights).

## Roll back

`git revert` the GitOps commit and reconcile. Images are pinned by digest, so a
revert restores the previous image if its tag still exists in
`127.0.0.1:5000`. Registry clean-ups keep only digests referenced by GitOps
manifests and Dockerfiles, digests running in the cluster, and the newest
image per repo. Check with `curl -s 127.0.0.1:5000/v2/<repo>/tags/list` before
relying on an older image. To restore Strata/Swift, change config only: the
runtime is in the image, and the weights remain in
`/data/models/qwen3.8/local/swift-1.5-flash-heretic-gyro-m`.
