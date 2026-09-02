#!/usr/bin/env bash
# Build the PR #36783 overlay image. CPU only -- never touches the GPU.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE="${BASE:-lmsysorg/sglang:nightly-dev-cu13-20260828-daf63171}"
TAG="${TAG:-sglang-hybrid-mtp-ngram:pr36783}"
PR_HEAD="63ad1158953dd92a665eda71e21a45f072d2bfd5"
CTXDIR="$HERE/pr"

if [[ ! -d "$CTXDIR/python/sglang" ]]; then
  echo "PR worktree missing. Recreate it with:" >&2
  cat >&2 <<EOF
  git clone --filter=blob:none --no-checkout https://github.com/shl518/sglang.git $HERE/fork
  cd $HERE/fork
  git remote add upstream https://github.com/sgl-project/sglang.git
  git fetch --filter=blob:none upstream main
  git fetch origin feature/hybrid-mtp-ngram
  git worktree add -f ../pr $PR_HEAD
EOF
  exit 1
fi

HAVE=$(git -C "$CTXDIR" rev-parse HEAD)
[[ "$HAVE" == "$PR_HEAD" ]] || {
  echo "worktree is at $HAVE, expected the PR head $PR_HEAD" >&2; exit 1; }

docker image inspect "$BASE" >/dev/null 2>&1 || docker pull "$BASE"

echo "building $TAG  (base $BASE, PR head ${PR_HEAD:0:9})"
docker build \
  --build-arg "BASE=$BASE" \
  -f "$HERE/Dockerfile.sglang-trial" \
  -t "$TAG" \
  "$CTXDIR"

echo
echo "built $TAG"
docker images "${TAG%%:*}" --format '  {{.Repository}}:{{.Tag}}  {{.Size}}'
echo "now run:  $HERE/run-sglang.sh hybrid"
