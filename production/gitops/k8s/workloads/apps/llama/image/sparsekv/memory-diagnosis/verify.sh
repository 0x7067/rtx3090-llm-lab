#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$(git rev-parse --show-toplevel)"
scratch=$(mktemp -d)
finish() {
  result=$?
  printf '%s\n' "$result" > "$here/checks.exit"
  rm -r -- "$scratch"
}
trap finish EXIT
scripts/test.sh
scripts/check-supply-chain.sh
kubectl kustomize k8s/workloads > "$scratch/workloads.yaml"
kubectl kustomize k8s/infra > "$scratch/infra.yaml"
scripts/check-supply-chain.sh --rendered-workloads "$scratch/workloads.yaml"
git diff --check
