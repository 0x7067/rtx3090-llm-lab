#!/usr/bin/env bash
# Build the orcasaq2-vllm image whose `gpu-profile` init container sets the
# GPU power limit and clocks. Recovered from GitOps scripts/build-orcasaq-image.sh
# (deleted in GitOps f13dde9d); the build context is this directory.
# Usage: build.sh [name:tag] [--publish]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gitops="$(cd "$here/../../gitops" && pwd)"
image=${1:-orcasaq-swap:v1}

docker build -t "$image" "$here"

case "${2:-}" in
  --publish) "$gitops/scripts/publish-image-to-k3s.sh" "$image" ;;
  "") ;;
  *) echo "usage: $0 [name:tag] [--publish]" >&2; exit 2 ;;
esac
