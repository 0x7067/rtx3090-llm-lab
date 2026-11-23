#!/usr/bin/env bash
# Stage the pinned portable runtime on Buttercup and the verified weights on the SSD.
set -Eeuo pipefail

image=vllm/vllm-openai:nightly-f8b29fab414884206460fe63b983ca34fc6e3e9b@sha256:3dfcb2ba18f41fce7400d643e52210a6c4f198019ead77146e9fb649ee600ceb
fork=https://github.com/Hiutaky/vllm.git
fork_revision=6e9d1cce3eadb9844d491de9b65dba24b9fec566
model=thinkingmachines/Inkling-2-Mini
model_revision=ed40e7da625f594fc9c4990f5f66be24873c22d7
runtime_parent=/data/models/inkling-2/vllm-ampere
runtime="$runtime_parent/6e9d1cce3ead-f8b29fab4"
model_parent=/data/development/llama-serving/inkling-2-mini
weights="$model_parent/vllm-$model_revision"
if (( $# != 0 )); then
    printf 'Usage: %s\n' "$0" >&2
    exit 2
fi

test "$(findmnt -nr -o TARGET -T /data/models)" = /data/buttercup_6tb
mountpoint -q /data/development
mkdir -p "$runtime_parent"
sudo -n install -d -m 0755 "$model_parent"
exec 9>"$runtime_parent/.stage.lock"
flock 9

bundle=
source=
staging=
cleanup() {
    local directory
    for directory in "$bundle" "$source" "$staging"; do
        [[ -d "$directory" ]] || continue
        sudo -n rm -r -- "$directory" </dev/null
    done
}
trap cleanup EXIT

verify_model() {
    hf cache verify "$model" --revision "$model_revision" --local-dir "$1" --fail-on-missing-files
}

if [[ ! -d "$runtime" ]]; then
    docker_store=$(sudo -n docker info --format '{{.DockerRootDir}}')
    docker_device=$(findmnt -nr -o MAJ:MIN -T "$docker_store")
    runtime_device=$(findmnt -nr -o MAJ:MIN -T "$runtime_parent")
    if [[ "$docker_device" != "$runtime_device" ]]; then
        printf 'Stage with the Docker image store on the same filesystem as %s.\n' "$runtime_parent" >&2
        exit 1
    fi
    bundle=$(mktemp -d "$runtime_parent/.runtime.XXXXXXXX")
    source=$(mktemp -d "$runtime_parent/.source.XXXXXXXX")
    git init -q "$source"
    git -C "$source" fetch -q --depth=1 "$fork" "$fork_revision"
    git -C "$source" checkout -q --detach FETCH_HEAD
    test "$(git -C "$source" rev-parse HEAD)" = "$fork_revision"
    sudo -n docker pull "$image"
    sudo -n docker run --rm --pull never -i --network none --user "$(id -u):$(id -g)" \
        --mount "type=bind,src=$(realpath "$bundle"),dst=/bundle" \
        --mount "type=bind,src=$(realpath "$source/vllm"),dst=/fork,readonly" \
        --entrypoint /bin/bash "$image" -se <<'EXTRACT'
set -Eeuo pipefail
copied=(/usr/bin/python3.14 /usr/lib/python3.14 /usr/include/python3.14
        /usr/include/x86_64-linux-gnu/python3.14
        /usr/local/lib/python3.14/dist-packages /usr/local/cuda-13.2)
for item in "${copied[@]}"; do
    printf 'Staging %s\n' "$item"
    cp -a --parents "$item" /bundle
done
ln -s python3.14 /bundle/usr/bin/python3
ln -s python3.14 /bundle/usr/bin/python
python3 - <<'PY'
from pathlib import Path
from shutil import copy2
source = Path("/fork")
target = Path("/bundle/usr/local/lib/python3.14/dist-packages/vllm")
for file in source.rglob("*"):
    if file.is_file() and file.suffix in {".py", ".json", ".yaml", ".yml", ".jinja"}:
        dest = target / file.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        copy2(file, dest)
PY
EXTRACT
    printf '%s\n%s\n' "$image" "$fork_revision" >"$bundle/provenance"
    chmod -R a+rX "$bundle"
    (cd "$bundle"; find usr -type f -print0 | sort -z | xargs -0 sha256sum > runtime.sha256)
    sudo -n chown -hR root:root "$bundle"
    sudo -n chmod 0755 "$bundle"
    sudo -n mv -T "$bundle" "$runtime"
else
    diff -u <(printf '%s\n%s\n' "$image" "$fork_revision") "$runtime/provenance"
    (cd "$runtime"; sha256sum --quiet --check runtime.sha256)
fi

if [[ ! -d "$weights" ]]; then
    staging=$(sudo -n mktemp -d "$model_parent/.model.XXXXXXXX")
    sudo -n chown "$(id -u):$(id -g)" "$staging"
    hf download "$model" --revision "$model_revision" --local-dir "$staging"
    verify_model "$staging"
    sudo -n chmod -R a+rX "$staging"
    sudo -n chown -hR root:root "$staging"
    sudo -n mv -T "$staging" "$weights"
fi
verify_model "$weights"
printf 'vLLM runtime: %s\nModel weights: %s\n' "$runtime" "$weights"
