#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
[[ ! -e $here/.scratch ]] || { echo 'Existing candidate scratch recorded in .scratch' >&2; exit 1; }
scratch=$(mktemp -d /var/tmp/sparsekv-build.XXXXXXXX)
trap 'rm -rf -- "$scratch"' EXIT

checkout() {
  local url=$1 commit=$2 target=$3
  git init -q "$target"
  git -C "$target" fetch -q --depth=1 "$url" "$commit"
  git -C "$target" checkout -q --detach FETCH_HEAD
  [[ $(git -C "$target" rev-parse HEAD) == "$commit" ]]
}
checkout https://github.com/bradj00/sparsekv-hyperqwen \
  29595e8712520ef017c51bcccb9ab6dbc3c68a01 "$scratch/sparse"
checkout https://github.com/syv-ai/HyperQwen \
  ce9615c4f2309be727407918a6c979bd67968069 "$scratch/hq"

git -C "$scratch/hq" apply --check "$scratch/sparse/patches/hyperqwen-base.diff"
git -C "$scratch/hq" apply "$scratch/sparse/patches/hyperqwen-base.diff"
cp -a "$scratch/sparse"/{sparsekv,serve-sparse.sh} "$scratch/hq/"
cp -a "$scratch/sparse"/pthfa/python/pth_flash_attn/*.py "$scratch/hq/pthfa/python/pth_flash_attn/"
cp -a "$scratch/sparse"/patches/sparse-*.patch "$scratch/hq/patches/"
(
  cd "$scratch/hq"
  echo 'dee8b28281d5837ad512a560e4d5c31af245cbe42dc1dea66dc1510775b4b99b  sparsekv/basis/basis27.pt' | sha256sum -c -
  ln -s /app/venv venv
)
printf '%s\n' "$scratch" > "$here/.scratch"
trap - EXIT
printf '%s\n' "$scratch"
