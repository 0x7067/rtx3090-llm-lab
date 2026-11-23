#!/usr/bin/env bash
# Download as the current user and publish verified, read-only serving assets with sudo.
set -euo pipefail

model_repo=WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3
model_revision=780140010a6c28f4f3c59a77c754ef8ea6542f01
model=Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer
model_sha=3701f1bf73d5f4ae3dd37e57c1a80817f2860f1c12b7d6de8815e1faf05b5a49
runtime_image=ghcr.io/iamwavecut/ninfer-all:cuda12.9@sha256:57dd28a7ece5bbb3bc5e6b9281f5be5d283902ee0c5e848ddc4d10b923e39011
stage=/data/models/inkling-2/ninfer-v3
production=/data/development/llama-serving/inkling-2-mini
destination="$production/780140010a6c-ninfer-2026-11-21.2"

if [[ $(findmnt -n -o TARGET -T /data/models) != /data/buttercup_6tb ]]; then
  printf '/data/models must resolve to the mounted Buttercup model library\n' >&2
  exit 1
fi
mountpoint -q /data/development
mkdir -p "$stage"
exec 9>"$stage/.stage.lock"
flock 9

verify_model() {
  printf '%s  %s\n' "$model_sha" "$1/$model" | sha256sum --check
}

if [[ -d "$destination" ]]; then
  verify_model "$destination"
  test "$(cat "$destination/runtime-image")" = "$runtime_image"
  (cd "$destination" && sha256sum --quiet --check runtime.sha256)
  printf 'Verified existing serving assets at %s\n' "$destination"
  exit 0
fi

hf download "$model_repo" "$model" --revision "$model_revision" --local-dir "$stage"
verify_model "$stage"
runtime=$(mktemp -d "$stage/.runtime.XXXXXX")
bundle=
trap 'rm -r -- "$runtime"; if [[ -n "$bundle" ]]; then sudo rm -r -- "$bundle"; fi' EXIT

docker run --rm -i --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$runtime,dst=/bundle" \
  --entrypoint /bin/bash "$runtime_image" -se <<'EXTRACT'
set -euo pipefail
source=/opt/ninfer/multi
mkdir -p /bundle/backends /bundle/lib
cp "$source/ninfer-serve" /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 /bundle/
cp -a "$source/backends/cuda12" /bundle/backends/
cp -a /opt/ninfer/licenses /bundle/
# Keep Debian's loader and userspace libraries private to the NInfer worker.
# The NVIDIA runtime supplies libcuda.so.1 from the host driver.
{
  ldd "$source/ninfer-serve"
  ldd "$source/backends/cuda12/libninfer-runtime.so"
} | awk '$2 == "=>" && $3 ~ "^/(usr/)?lib/" {print $3}' | sort -u |
  xargs -r cp -L -t /bundle/lib
EXTRACT

# Match the serving filesystem's reserve used for previous Inkling bundles.
sudo install -d -m 0755 "$production"
required=$((32 * 1024**3 + $(stat -c '%s' "$stage/$model") + $(du -sb "$runtime" | cut -f1)))
available=$(df --output=avail -B1 "$production" | tail -1)
if ((available < required)); then
  printf 'Inkling stage needs %s bytes including SSD reserve; %s available\n' "$required" "$available" >&2
  exit 1
fi

bundle=$(sudo mktemp -d "$production/.inkling.XXXXXX")
sudo chmod 0755 "$bundle"
sudo cp -a --no-preserve=ownership "$runtime" "$bundle/runtime"
sudo chmod -R a+rX "$bundle/runtime"
sudo install -m 0644 "$stage/$model" "$bundle/$model"
printf '%s\n' "$runtime_image" | sudo tee "$bundle/runtime-image" >/dev/null
verify_model "$bundle"
(
  cd "$bundle"
  find runtime -type f -print0 | sort -z | xargs -0 sha256sum
) | sudo tee "$bundle/runtime.sha256" >/dev/null
sudo mv -T "$bundle" "$destination"
bundle=
printf 'Verified serving assets staged at %s\n' "$destination"
