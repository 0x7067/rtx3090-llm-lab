# ocr

Finance OCR (PaddleOCR-VL-1.6 + PP-DocLayoutV3) served as the `paddleocr-vl` llama-swap
model, so it shares the 3090 with the text and image models: a request swaps the card to
it, and `ttl: 900` hands it back.

This directory is self-contained: the reviewed OCR source (`services/`, `shared/`,
`package*.json`), the pinned dependency locks, the image recipe and the launcher. Only
the weights live outside, in the model library (`/data/buttercup_6tb/k3s/llama-models/ocr/`,
mounted at `/models/ocr` and `/models/layout`). `evaluation/` holds the synthetic
fixtures and ground truth. The service's own docs are `services/ocr/README.md` and
`services/ocr/deploy/README.md`.

## What runs

`llama-swap` starts `/opt/ocr/launch.py`, which starts, in one process group each:

1. vLLM 0.10.2 with PaddleOCR-VL-1.6 (the reviewed `run_vllm.py`) on `127.0.0.1:8000`.
   `hostAliases` points `ocr-vllm` at loopback because the manifest pins that URL.
2. The durable OCR API (`services/ocr/main.ts`, SQLite queue in `/ocr-data`) on `127.0.0.1:9417`.
3. A front on `${PORT}` that answers `GET /health` (503 until vLLM is up and the API listens)
   and forwards every other connection to the API.

The pinned preflight requires Python 3.12.15, but the base image has 3.12.3, so the
image carries 3.12.15 under `/opt/py312` with two venvs (`venv-layout`, `venv-vllm`).
`LD_LIBRARY_PATH` for it is set by the launcher only, never image-wide.

## Calling it

Two tokens: the llama bearer token (llama-swap) and the OCR service token (Secret
`ocr-service`, key `token`). llama-swap and the OCR API both read `Authorization`, so
send the llama token there and the OCR token as `X-OCR-Token`; the launcher's front
moves it into `Authorization` before the request reaches the API, which still does its
own authentication. The API is the OCR contract v1, not an OpenAI API:

    POST /upstream/paddleocr-vl/v1/ocr/jobs     # multipart: metadata + original, 202 + job id
    GET  /upstream/paddleocr-vl/v1/ocr/jobs/<id> # poll

The first request after a swap waits for vLLM to load; llama-swap's `healthCheckTimeout`
(900s) covers it.

## Constraints inherited from the service

- One dedicated vLLM, one document at a time, 300s per job; the manifest rejects any change
  to the reviewed profile. Jobs are durable; a swap-out mid-job is recovered as
  `worker_interrupted`, so do not let other models preempt a queued job (poll through
  llama-swap, which keeps the idle clock running).
- **Synthetic documents only** until the evaluation in `services/ocr/README.md` ("Pending
  acceptance") has been done. Namespace is `finance-ocr-synthetic-trial`.
- The NetworkPolicy (`../networkpolicy.yaml`) keeps `:8000` and `:9417` inside the pod.

## Build and deploy

    cd k8s/workloads/apps/llama/ocr
    H=$(python3 services/ocr/deploy/source_inventory.py)
    docker build -f Dockerfile.llama-swap -t 127.0.0.1:5000/home-llm-ocr:vN \
      --build-arg BASE=<current home-llm-qwen-optimized digest> --build-arg OCR_ADAPTER_TREE_SHA256=$H .
    scripts/publish-image-to-k3s.sh home-llm-ocr:vN        # prints the digest
    python3 make_manifest.py 127.0.0.1:5000/home-llm-ocr@sha256:<digest>   # writes ../ocr-manifest.yaml
    # set that digest as the llama Deployment image, commit, push, flux reconcile

The image is `home-llm-qwen-optimized` plus the OCR layer, so it replaces that image in
the Deployment and every other model keeps working. The manifest embeds the image
reference, so regenerate it for every rebuild.

## Photo-processing fix — 24 October 2026

The authorized finance-app update keeps the deployed manager/launcher, pinned models,
dependencies, credentials, limits, queue settings and mounts. It replaces only nine
reviewed OCR source/template files with the bounded camera-photo preparation and
normalization-v4 fixes from finance-ocr commit 5d658e0. All original bytes/evidence and
human review remain required; this is not automatic financial approval.

A source-only image layer was assembled using the OCI image/distribution format and
published to the existing loopback registry. The Docker publish script cannot publish
this artifact without first importing the 35.8 GiB base into a second image store; the
one-off additive publication avoids that shared-disk load. All 113 base descriptors
and DiffIDs are retained exactly, and no RUN/dependency resolution/model download
occurs. The new source layer is 33,826 compressed bytes. The final config/manifest/blob
digests were verified against the registry. This is an operational exception for this
source-only repair; the ordinary build/publish workflow above is unchanged.

`photo-v4-release.json` records the exact base, candidate, source inventory and layer
identity. The runtime manifest is regenerated with the existing generator; only image
references, source hash and normalization revision change. Models, package locks,
launcher and other runtime settings remain identical. The old image and manifest
remain the rollback reference. The actual candidate pod passed source/package/parser
and pipeline checks without GPU access. Worker37/deployment19 tests, supply-chain
check, both Kustomization renders, rendered-workload policy and policy unit tests pass.
Live synthetic and authorized private receipt comparison follows the controlled rollout;
these offline checks do not certify real-photo quality or sustained host reliability.
