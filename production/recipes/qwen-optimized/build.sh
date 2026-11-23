#!/usr/bin/env bash
set -euo pipefail

image=${1:-home-llm-qwen-optimized:20261021-v1}
source_checkout=${2:-/tmp/strata-research-20261021}
engine_archive=${3:-/data/qwen-candidates-20261021/runtime/strata-engine.tar.gz}
source_revision=011c9e600a93bbc9fd451c563501b5831c32bbca
engine_sha256=45a6df98aaf4b6b7c05087a4ab428be8df478ddcc07193a13cbaab312b85895e
here=$(cd "$(dirname "$0")" && pwd)
context=$(mktemp -d /tmp/qwen-optimized-image.XXXXXX)
trap 'rm -rf "$context"' EXIT

if ! test -d "$source_checkout/.git"; then
  source_checkout="$context/checkout"
  git init "$source_checkout"
  git -C "$source_checkout" fetch --depth 1 https://github.com/Niko1221/Strata "$source_revision"
fi
if ! test -f "$engine_archive"; then
  engine_archive="$context/engine.tar.gz"
  curl --fail --location --retry 3 \
    https://github.com/Niko1221/Strata/releases/download/v0.2.2/strata-engine-0.1.47-linux-x86_64-cuda-cuda13.tar.gz \
    --output "$engine_archive"
fi
printf '%s  %s\n' "$engine_sha256" "$engine_archive" | sha256sum --check --status
mkdir "$context/source" "$context/engine"
git -C "$source_checkout" archive "$source_revision" | tar -x -C "$context/source"
tar -xzf "$engine_archive" -C "$context/engine"
cp "$here/Dockerfile" "$context/Dockerfile"
printf 'checkout\nengine.tar.gz\n' > "$context/.dockerignore"
docker build --tag "$image" "$context"
