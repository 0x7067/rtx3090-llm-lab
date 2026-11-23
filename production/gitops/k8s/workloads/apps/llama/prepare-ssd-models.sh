#!/usr/bin/env bash
# Stage the read-only inference assets used by deployment.yaml on the root NVMe.
set -euo pipefail

destination=/srv/llama-nvme/production-models
text_source=/data/models/qwen3.8/local/hyperqwen-bakeoff/models
asset_source=/data/buttercup_6tb/k3s/llama-models
text_models=(
  Qwen3.8-27B-W4A16-AutoRound
)
assets=(
  qwen3.8/harness-hub/Qwen3.8-27B-pi-adapted/3cad999e2b9ec72fea8d802cc286e4502adfa082
  qwen-image
  glm-ocr-api/406ab9bc91fe/source
  glm-ocr-sdk-runtime/b1116991fea7/runtime
  glm-ocr-baseline/65a42de1148d
  ocr
)
[[ $EUID == 0 ]]
[[ $(findmnt -n -T /srv/llama-nvme -o SOURCE) == /dev/mapper/nvme--vg-ubuntu--lv ]]
mountpoint -q /data/buttercup_6tb
[[ ! -L $destination ]]
install -d -m 0755 "$destination"
rm -f "$destination/.staged-at"

sources=() targets=()
for model in "${text_models[@]}"; do
  sources+=("$text_source/$model")
  targets+=("$destination/text/$model")
done
for asset in "${assets[@]}"; do
  sources+=("$asset_source/$asset")
  targets+=("$destination/$asset")
done

# Account for a previous partial copy without charging the space twice.
required=$((32*1024**3))
for i in "${!sources[@]}"; do
  src=${sources[i]} dst=${targets[i]}
  test -d "$src"
  install -d -m 0755 "$dst"
  read -r source_bytes _ < <(du -sx --block-size=1 "$src")
  read -r staged_bytes _ < <(du -sx --block-size=1 "$dst")
  if ((source_bytes > staged_bytes)); then
    required=$((required+source_bytes-staged_bytes))
  fi
done
available=$(df --output=avail -B1 "$destination" | tail -1)
if ((available < required)); then
  echo "Production stage needs $required bytes including SSD reserve; $available available" >&2
  exit 1
fi

for i in "${!sources[@]}"; do
  src=${sources[i]} dst=${targets[i]}
  [[ $(findmnt -n -T "$dst" -o SOURCE) == /dev/mapper/nvme--vg-ubuntu--lv ]]
  printf 'Copying %s at %s\n' "${dst#"$destination"/}" "$(date -Is)"
  # rsync verifies each transferred file. Match nanosecond timestamps and
  # preserve data, links, permissions, ACLs and extended attributes.
  ionice -c 2 -n 5 rsync -aHAXxS --numeric-ids --modify-window=-1 --info=stats1 "$src/" "$dst/"
  differences=$(rsync -aHAXxSni --numeric-ids --modify-window=-1 --delete "$src/" "$dst/")
  if [[ -n $differences ]]; then
    printf 'Staged files differ from %s:\n%s\n' "$src" "$differences" >&2
    exit 1
  fi
done
date -Is > "$destination/.staged-at"
