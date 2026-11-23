#!/usr/bin/env bash
set -euo pipefail
image=${1:-home-llm-gyro:20261016-v1}
here=$(cd "$(dirname "$0")" && pwd)
context=$(mktemp -d /tmp/gyro-image.XXXXXX)
test "$(docker image inspect local/strata-bakeoff:v0.1.44.1 --format '{{.Id}}')" = \
  sha256:ba010c9a39f0318ac9f0341ac2c1476fa1fcddb3bef37f128ec0f9357dbbe354
git clone --depth 1 --branch rc2 https://github.com/agentionai/Strata "$context/strata"
git -C "$context/strata" fetch --depth 1 origin 9ff02c9fa9c86a88088370925b3c9131cc5d6c26
git -C "$context/strata" checkout --detach FETCH_HEAD
git clone --depth 1 https://gitlab.com/agentionai-group/llama.cpp "$context/strata/gyro-llama"
git -C "$context/strata/gyro-llama" fetch --depth 1 origin 9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f
git -C "$context/strata/gyro-llama" checkout --detach FETCH_HEAD
cp "$here/Dockerfile" "$context/strata/Dockerfile.gyro"
printf '.git\ngyro-llama/.git\n' > "$context/strata/.dockerignore"
BUILDX_CONFIG=${BUILDX_CONFIG:-/tmp/gyro-buildx} docker build \
  -f "$context/strata/Dockerfile.gyro" -t "$image" "$context/strata"
