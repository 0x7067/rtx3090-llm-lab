#!/usr/bin/env bash
# Sync production/ from GitOps main and push the result to this repo's main.
#
# Runs from a dedicated clone of this repo, never from a working checkout: the
# rtx3090-lab-sync wrapper in ~/dotfiles resets that clone to origin/main and
# then runs this script. Needs read access to Denguinho/docker-services,
# push access to this repo, and passwordless sudo for host inputs the user
# cannot read.
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
