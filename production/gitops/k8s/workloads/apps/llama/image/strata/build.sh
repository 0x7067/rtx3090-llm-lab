#!/usr/bin/env bash
# Usage: build.sh [tag]. Fetches the pinned Strata v0.2.8 source and engine, verifies
# both, builds with BuildKit and pushes to the local registry, printing the digest ref.
# Plain `docker build` fails on this base ("max depth exceeded"); see the Dockerfile.
set -euo pipefail

tag=${1:-0.2.8-v1}
image=127.0.0.1:5000/home-llm-strata:$tag
source_revision=870aedd9e1ebe1c91f94da93b59af17729f6c5c0   # tag v0.2.8
engine_url=https://github.com/Niko1221/Strata/releases/download/v0.2.8/strata-engine-0.1.54-linux-x86_64-cuda-cuda13.tar.gz
engine_sha256=a183c298e26145a11b87baaeda5fe2ec32b305c600e6886206fd225bf3070726
here=$(cd "$(dirname "$0")" && pwd)
context=$(mktemp -d "${TMPDIR:-/tmp}/strata-image.XXXXXX")
trap 'rm -rf "$context"' EXIT

git init -q "$context/checkout"
git -C "$context/checkout" fetch -q --depth 1 https://github.com/Niko1221/Strata "$source_revision"
mkdir "$context/source" "$context/engine"
git -C "$context/checkout" archive "$source_revision" | tar -x -C "$context/source"
curl --fail --location --retry 3 --silent --show-error "$engine_url" --output "$context/engine.tar.gz"
printf '%s  %s\n' "$engine_sha256" "$context/engine.tar.gz" | sha256sum --check --status
tar -xzf "$context/engine.tar.gz" -C "$context/engine"
uv pip install --quiet --require-hashes --no-deps --only-binary :all: --python-version 3.12 \
  --target "$context/site" -r "$here/requirements.txt"
cp "$here/Dockerfile" "$context/"
printf 'checkout\nengine.tar.gz\n' > "$context/.dockerignore"

builder=strata-local-registry
if ! docker buildx inspect "$builder" >/dev/null 2>&1; then
  printf '[registry."127.0.0.1:5000"]\n  http = true\n' > "$context/buildkitd.toml"
  docker buildx create --name "$builder" --driver docker-container \
    --driver-opt network=host --buildkitd-config "$context/buildkitd.toml" >/dev/null
fi
docker buildx build --builder "$builder" --metadata-file "$context/meta.json" \
  --output "type=image,name=$image,push=true" "$context"
digest=$(jq -r '."containerimage.digest"' "$context/meta.json")
echo "published: reference it in the manifest as  image: $image@$digest"
