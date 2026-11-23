#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$here/../.."
benchmark_check_tmp=$(mktemp -d /tmp/sparsekv-repo-check.XXXXXX)
finish() {
  local status=$?
  rm -rf -- "$benchmark_check_tmp"
  printf '%s\n' "$status" > "$here/checks.exit"
}
trap finish EXIT
scripts/test.sh
scripts/check-supply-chain.sh
kubectl kustomize k8s/workloads > "$benchmark_check_tmp/workloads.yaml"
kubectl kustomize k8s/infra > "$benchmark_check_tmp/infra.yaml"
scripts/check-supply-chain.sh --rendered-workloads "$benchmark_check_tmp/workloads.yaml"
bash -n "$here"/*.sh "$here"/harness/*.sh
shellcheck "$here"/*.sh "$here"/harness/*.sh
