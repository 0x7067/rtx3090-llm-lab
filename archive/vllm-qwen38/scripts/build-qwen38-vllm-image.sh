#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image=${1:-qwen38-27b-3090:v7}
base_image=${QWEN38_BASE_IMAGE:-qwen38-27b-3090:v4}
context="$repo_root/rtx3090-llm-lab/vllm/image-build/overlay"

docker build \
  --build-arg "BASE_IMAGE=$base_image" \
  -t "$image" \
  "$context"

case "${2:-}" in
  --import|--publish)
    # Both go through the host-local registry now (no sudo); reference the
    # result as 127.0.0.1:5000/$image in the manifest.
    "$repo_root/scripts/publish-image-to-k3s.sh" "$image"
    ;;
  "") ;;
  *)
    echo "usage: $0 [name:tag] [--import|--publish]" >&2
    exit 2
    ;;
esac
