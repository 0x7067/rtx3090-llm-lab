#!/usr/bin/env bash
set -euo pipefail
[[ ${1:-} == --execute && ( $# == 2 || $# == 3 ) ]] || {
  echo "Usage: $0 --execute NEW_OUTPUT_DIR [COMPLETED_BASELINE_RUN]" >&2
  exit 2
}
here=$(cd -- "$(dirname -- "$0")" && pwd)
out=$(realpath -m -- "$2")
reuse=${3:-}
runtime=${SPARSEKV_RUNTIME_SECONDS:-3600}
[[ $runtime =~ ^[0-9]+$ && $runtime -ge 600 && $runtime -le 3600 ]] || exit 2
if [[ -n $reuse ]]; then
  reuse=$(realpath -- "$reuse")
  [[ $(cat "$reuse/baseline-exit") == 0 && -f $reuse/baseline/long-context-summary.json ]] || exit 2
fi
for path in "$here" "$out"; do
  [[ $path =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo "Unsupported path: $path" >&2; exit 2; }
done
[[ $(id -un) == denguinho ]] || { echo "Run as denguinho" >&2; exit 2; }
"$here/orchestrate.sh" --preflight
mkdir -m 700 -- "$out"
exec sudo -n systemd-run --collect --unit sparsekv-production-bench \
  --description 'Compare production HyperQwen and SparseKV and restore production' \
  --uid="$(id -u)" --gid="$(id -g)" --working-directory="$here" \
  --setenv="PATH=$PATH" --setenv="XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" \
  --setenv=SPARSEKV_BENCH_UNIT=sparsekv-production-bench \
  --setenv="KUBECONFIG=${KUBECONFIG:-/home/denguinho/.kube/config}" \
  --setenv="SPARSEKV_REUSE_BASELINE=$reuse" \
  --property="RuntimeMaxSec=${runtime}s" --property=TimeoutStopSec=600s \
  --property="ExecStopPost=$here/orchestrate.sh --restore $out" \
  "$here/orchestrate.sh" --execute "$out"
