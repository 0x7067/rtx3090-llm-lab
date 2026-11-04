#!/usr/bin/env bash
# Run a command with exclusive use of the RTX 3090: suspend Flux, scale the
# llama-swap API to zero, and restore both on success, failure, or TERM.
# Extracted from resume-regeneration.sh, which proved the sequence over a
# 13-hour run. Holds the same lock, so a training job and a capture job
# cannot both claim the card.
#
# Usage: with-local-api-paused.sh <command> [args...]
set -euo pipefail
(( $# > 0 )) || { echo 'Usage: with-local-api-paused.sh <command> [args...]' >&2; exit 2; }
W=/data/buttercup_6tb/specforge-work
exec 9>"$W/.k2-training.lock"
flock -n 9 || { echo 'Another K2 training job owns the lock' >&2; exit 1; }
# User services inherit neither the interactive kubeconfig nor its local bin.
export PATH="$HOME/.local/bin:$PATH"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
for command in flux kubectl nvidia-smi; do
  command -v "$command" > /dev/null || { echo "Missing command: $command" >&2; exit 1; }
done
replicas=$(kubectl -n apps get deploy llama -o jsonpath='{.spec.replicas}')
suspended=$(kubectl -n flux-system get kustomization apps -o jsonpath='{.spec.suspend}')
[[ "$replicas" == 1 && "$suspended" != true ]] || {
  echo 'Expected the normal one-replica API with Flux apps active; another operation may own it' >&2
  exit 1
}
restore=0
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  set +e
  if [[ "$restore" == 1 ]]; then
    kubectl -n apps scale deploy/llama --replicas="$replicas" || rc=1
    flux resume kustomization apps || rc=1
    kubectl -n apps rollout status deploy/llama --timeout=15m || rc=1
  fi
  echo "Command finished with exit=$rc; local API restoration attempted"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
restore=1
flux suspend kustomization apps
kubectl -n apps scale deploy/llama --replicas=0
kubectl -n apps wait --for=delete pod -l app.kubernetes.io/name=llama --timeout=5m
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "$used" -lt 500 ]] || { echo "GPU remains occupied: ${used} MiB" >&2; exit 1; }
"$@"
