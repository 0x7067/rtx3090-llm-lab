#!/usr/bin/env bash
set -euo pipefail

# CUDA llama.cpp (server + bench) plus the llama-swap proxy, for the unified
# `llama` app. Build source is the rtx3090-llm-lab submodule (Dockerfile at its
# root, patches-v14/). The image is pushed to the host-local registry that k3s
# pulls from (no sudo, no containerd import); bump the tag on every rebuild and
# reference it in deployment.yaml as 127.0.0.1:5000/<IMAGE_NAME>.
IMAGE_NAME="${IMAGE_NAME:-llama:cuda-swap-v14}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_DIR="${ROOT_DIR}/rtx3090-llm-lab"

docker build -t "${IMAGE_NAME}" "${IMAGE_DIR}"
"${ROOT_DIR}/scripts/publish-image-to-k3s.sh" "${IMAGE_NAME}"
