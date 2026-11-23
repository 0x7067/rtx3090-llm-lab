#!/usr/bin/env bash
# Sync production/ from GitOps main and push the result to this repo's main.
#
# Run by hand on denguinho-server after a promotion, from a checkout of this
# repo at origin/main with a clean tree (it refuses otherwise). Needs read
# access to Denguinho/docker-services, push access to this repo, and
# passwordless sudo for host inputs the user cannot read.
set -Eeuo pipefail

lab="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
state=${RTX3090_LAB_SYNC_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/rtx3090-lab-sync}
gitops_url=${GITOPS_URL:-git@github.com:Denguinho/docker-services.git}
gitops="$state/docker-services.git"

if [[ ! -d $gitops ]]; then
  git clone --quiet --bare "$gitops_url" "$gitops"
fi
git -C "$gitops" fetch --quiet origin '+refs/heads/main:refs/heads/main'

cd "$lab"
git fetch --quiet origin main
if [[ -n $(git status --porcelain) ]]; then
  echo "publish.sh: working tree is not clean; commit or stash first" >&2; exit 1
fi
if [[ $(git rev-parse HEAD) != $(git rev-parse FETCH_HEAD) ]]; then
  echo "publish.sh: HEAD is not origin/main; run: git checkout --detach origin/main" >&2; exit 1
fi
"$lab/tools/production/sync.py" sync --gitops "$gitops" --ref main --host-scan
"$lab/tools/production/sync.py" verify
gitleaks dir production --config .gitleaks.toml --redact --no-banner --log-level warn

if [[ -z $(git status --porcelain -- production) ]]; then
  echo "production/ already matches GitOps $(git -C "$gitops" rev-parse --short=12 main)"
  exit 0
fi
git add -A -- production
git commit --quiet -m "production: sync GitOps $(git -C "$gitops" rev-parse --short=12 main)"
git push --quiet origin HEAD:main
echo "pushed $(git rev-parse --short=12 HEAD)"
