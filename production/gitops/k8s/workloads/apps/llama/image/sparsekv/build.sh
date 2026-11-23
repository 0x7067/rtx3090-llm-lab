#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$here"
mkdir -p qualification
publish_cache=''
builder_started=0
builder_registered=0
finish() {
  result=$?
  if (( builder_registered )); then docker buildx rm sparsekv-adoption || result=1; fi
  if (( builder_started )); then docker stop sparsekv-adoption-builder || result=1; fi
  if [[ -n $publish_cache ]]; then sudo -n rm -r -- "$publish_cache" || result=1; fi
  printf '%s\n' "$result" > qualification/build.exit
  exit "$result"
}
trap finish EXIT
[[ ! -e qualification/build-metadata.json ]] || { echo 'Already published; use a new tag for a rebuild' >&2; exit 1; }
tag=127.0.0.1:5000/home-llm-strata:sparsekv-v1
base=127.0.0.1:5000/home-llm-strata@sha256:83077f60377d9e303d446a2f045381daab28a123e8e314c5bbe1ba53314884a2
http=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:5000/v2/home-llm-strata/manifests/sparsekv-v1")
[[ $http == 404 ]] || { echo "Tag preflight returned $http, expected 404" >&2; exit 1; }
[[ -d .source ]] || bash stage.sh
[[ ! -e .runtime && ! -e .runtime.tar ]] || { echo 'Inspect previous runtime output before retrying' >&2; exit 1; }
# The user owns the archive; sudo is scoped to the disposable containerd task.
# shellcheck disable=SC2024
sudo -n k3s ctr -n k8s.io run --rm --cpus 2 --memory-limit 4294967296 \
  --env CUDA_VISIBLE_DEVICES= --env MAX_JOBS=2 --env PIP_NO_INDEX=1 --env PIP_NO_CACHE_DIR=1 \
  --env PY=/opt/sparsekv/venv/bin/python \
  --mount "type=bind,src=$here/.source,dst=/source,options=rbind:ro" \
  --mount "type=bind,src=$here,dst=/recipe,options=rbind:ro" \
  "$base" sparsekv-adoption-build /bin/bash -euc '
    mkdir /opt/sparsekv
    cp -a /source/. /opt/sparsekv/
    cp /recipe/isolate_runtime.py /recipe/check_image.py /opt/sparsekv/
    /app/venv/bin/python -I -S /opt/sparsekv/isolate_runtime.py >&2
    cd /opt/sparsekv
    bash pthfa/install.sh >&2
    bash sparsekv/install.sh /opt/sparsekv/venv >&2
    /opt/sparsekv/venv/bin/python /opt/sparsekv/check_image.py >&2
    tar -C /opt -cf - sparsekv
  ' > .runtime.tar
sha256sum .runtime.tar > qualification/runtime-archive.sha256
mkdir .runtime
tar -xf .runtime.tar --strip-components=1 -C .runtime
publish_cache=$(mktemp -d /data/buttercup_6tb/sparsekv-publish.XXXXXXXX)
docker run -d --rm --name sparsekv-adoption-builder --privileged --network host \
  --cpus 2 --memory 5g --memory-swap 5g \
  --mount "type=bind,src=$publish_cache,dst=/var/lib/buildkit" \
  --mount "type=bind,src=$here/buildkitd.toml,dst=/etc/buildkit/buildkitd.toml,readonly" \
  moby/buildkit@sha256:cec9f139f45e93c5c69c60f8b07cfad9f43f4ef6b6a6cd917527fea5ff2e3dea \
  --addr tcp://127.0.0.1:12349 --config /etc/buildkit/buildkitd.toml
builder_started=1
docker buildx create --name sparsekv-adoption --driver remote tcp://127.0.0.1:12349
builder_registered=1
timeout 600 docker buildx build --builder sparsekv-adoption --network=none \
  --resource=memory=4g --resource=cpu-quota=200000 --progress=plain \
  --metadata-file qualification/build-metadata.json \
  --output "type=image,name=$tag,push=true" .
