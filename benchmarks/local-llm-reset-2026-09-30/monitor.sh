#!/usr/bin/env bash
set -euo pipefail

trial_container=$1
telemetry_path=$2

while [ "$(docker inspect "$trial_container" --format '{{.State.Running}}')" = true ]; do
  observed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  pressure_avg10=$(awk '$1 == "some" {split($2, value, "="); print value[2]}' /proc/pressure/memory)
  gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  container_memory=$(docker stats --no-stream --format '{{.MemUsage}}' "$trial_container")
  printf '%s\t%s\t%s\t%s\t%s\n' "$observed_at" "$available_kib" "$pressure_avg10" "$gpu_used_mib" "$container_memory" >> "$telemetry_path"
  if [ "$available_kib" -lt 12582912 ] || awk -v pressure="$pressure_avg10" 'BEGIN {exit !(pressure > 10)}'; then
    printf 'abort\t%s\tavailable_kib=%s\tpressure_avg10=%s\n' "$observed_at" "$available_kib" "$pressure_avg10" >> "$telemetry_path"
    docker stop -t 20 "$trial_container"
    exit 1
  fi
  sleep 5
done
