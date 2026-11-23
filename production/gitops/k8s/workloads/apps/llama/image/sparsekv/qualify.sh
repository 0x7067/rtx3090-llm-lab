#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
output=${1:?Usage: qualify.sh NEW_OUTPUT_DIRECTORY}
mkdir "$output"
finish() {
  result=$?
  printf '%s\n' "$result" > "$output/exit-code"
}
trap finish EXIT
kubectl -n apps rollout status deploy/llama --timeout=900s
kubectl -n apps get deploy llama -o json > "$output/deployment.json"
kubectl -n apps get pods -l app.kubernetes.io/name=llama -o json > "$output/pods-before.json"
# Preload can outlast the manager's readiness probe; this actual request is bounded.
timeout 1200 kubectl -n apps exec -i deploy/llama -c llama-swap -- \
  /app/venv/bin/python - < "$here/qualify_live.py" | tee "$output/live.jsonl"
kubectl -n apps get pods -l app.kubernetes.io/name=llama -o json > "$output/pods-after.json"
kubectl -n apps logs deploy/llama -c llama-swap --tail=300 > "$output/backend.log"
kubectl -n flux-system get kustomizations -o json > "$output/flux.json"
