#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# syv-ai stack vendored in the lab submodule (vllm/syv-ai, subtree at our k8s-deploy-v10 overlay)
source_root=${QWEN38_SOURCE_ROOT:-$repo_root/rtx3090-llm-lab/vllm/syv-ai}
lab_build="$repo_root/rtx3090-llm-lab/vllm/image-build"
image=${1:-qwen38-27b-3090:base-locked}
build_root=$(mktemp -d /tmp/qwen38-vllm-base.XXXXXX)

cleanup() {
  rm -rf -- "$build_root"
}
trap cleanup EXIT

test -f "$source_root/verify.sh"
cp -a "$source_root/." "$build_root/"
cp "$lab_build/base/requirements.lock" \
  "$build_root/docker/requirements.lock"
cp "$lab_build/base/Dockerfile.locked" \
  "$build_root/Dockerfile"
cp "$lab_build"/overlay/*.patch \
  "$build_root/patches/"

if [[ "${2:-}" == "--check" ]]; then
  docker build --check "$build_root"
  exit 0
fi

docker build -t "$image" "$build_root"

case "${2:-}" in
  --publish)
    "$repo_root/scripts/publish-image-to-k3s.sh" "$image"
    ;;
  ""|--check) ;;
  *)
    echo "usage: $0 [name:tag] [--check|--publish]" >&2
    exit 2
    ;;
esac
