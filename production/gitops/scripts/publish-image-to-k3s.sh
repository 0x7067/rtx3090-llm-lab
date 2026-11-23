#!/usr/bin/env bash
# Publish a locally built image to the host-local registry that k3s pulls from.
#
# No sudo needed: the `local-k3s-registry` container serves 127.0.0.1:5000 over
# plain HTTP, and containerd pulls from it directly when a manifest references
# `127.0.0.1:5000/<name>:<tag>` (verified 2026-09-02: kubelet pulled a 3.5 GB
# image in 228 ms with imagePullPolicy Always). So the flow is:
#
#   docker build -t llama:cuda-swap-v15 rtx3090-llm-lab/
#   scripts/publish-image-to-k3s.sh llama:cuda-swap-v15
#   # then use the digest-pinned image reference printed below in deployment.yaml
#
# Contract: stdout is exactly one line, `ref=<registry>/<name>:<tag>@sha256:<64 hex>`;
# all human-readable text goes to stderr. REGISTRY overrides 127.0.0.1:5000.
# Always bump the tag on a rebuild: IfNotPresent will not re-pull a same-tag image
# that containerd already holds.
set -euo pipefail
image=${1:?usage: publish-image-to-k3s.sh name:tag}
registry=${REGISTRY:-127.0.0.1:5000}
if [[ "$image" == */* || "$image" != *:* ]]; then
  echo "expected an unqualified name:tag, got $image" >&2
  exit 2
fi
remote_image="$registry/$image"
docker tag "$image" "$remote_image"
# Docker 29 can print the reference (rather than the digest) with --quiet.
# Read the digest from the successful push itself, never from a later tag lookup.
if ! push_output=$(docker push "$remote_image" 2>&1); then
  printf '%s\n' "$push_output" >&2
  exit 1
fi
if [[ "$push_output" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  published_digest=$push_output
elif [[ "$push_output" =~ digest:\ (sha256:[0-9a-f]{64}) ]]; then
  published_digest=${BASH_REMATCH[1]}
else
  echo "push succeeded but returned no valid digest; resolve it before deploying" >&2
  exit 1
fi
echo "published: reference it in the manifest as the ref below" >&2
echo "ref=$remote_image@$published_digest"
