#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
[[ ! -e $here/.source ]] || { echo '.source already exists; inspect it before restaging' >&2; exit 1; }
checkout=$(mktemp -d)
trap 'rm -r -- "$checkout"' EXIT
fetch() {
  git init -q "$3"
  git -C "$3" fetch -q --depth=1 "$1" "$2"
  git -C "$3" checkout -q --detach FETCH_HEAD
  [[ $(git -C "$3" rev-parse HEAD) == "$2" ]]
}
fetch https://github.com/bradj00/sparsekv-hyperqwen 29595e8712520ef017c51bcccb9ab6dbc3c68a01 "$checkout/sparse"
fetch https://github.com/syv-ai/HyperQwen ce9615c4f2309be727407918a6c979bd67968069 "$checkout/hq"
mkdir "$here/.source"
git -C "$checkout/hq" archive HEAD | tar -x -C "$here/.source"
patch --dry-run --fuzz=0 -p1 -d "$here/.source" < "$checkout/sparse/patches/hyperqwen-base.diff"
patch --fuzz=0 -p1 -d "$here/.source" < "$checkout/sparse/patches/hyperqwen-base.diff"
cp -a "$checkout/sparse"/{sparsekv,serve-sparse.sh} "$here/.source/"
cp -a "$checkout/sparse"/pthfa/python/pth_flash_attn/*.py "$here/.source/pthfa/python/pth_flash_attn/"
cp -a "$checkout/sparse"/patches/sparse-*.patch "$here/.source/patches/"
(cd "$here/.source"; echo 'dee8b28281d5837ad512a560e4d5c31af245cbe42dc1dea66dc1510775b4b99b  sparsekv/basis/basis27.pt' | sha256sum -c -)
printf '%s\n' 'SparseKV 29595e8712520ef017c51bcccb9ab6dbc3c68a01' 'HyperQwen ce9615c4f2309be727407918a6c979bd67968069' > "$here/.source/SOURCE_COMMITS"
