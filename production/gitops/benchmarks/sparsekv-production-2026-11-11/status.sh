#!/usr/bin/env bash
set -eu
here=$(cd -- "$(dirname -- "$0")" && pwd)
systemctl show sparsekv-production-bench -p ActiveState -p SubState -p Result
awk '/MemAvailable/ {printf "Available RAM: %.2f GiB\n", $2/1024/1024}' /proc/meminfo
kubectl -n apps get deployment llama -o custom-columns=NAME:.metadata.name,DESIRED:.spec.replicas,AVAILABLE:.status.availableReplicas
for state in "$here"/run-*/{baseline,candidate}/phases.tsv; do
  if [[ -f $state ]]; then
    printf '\n%s\n' "$state"
    tail -n 3 "$state"
  fi
done
