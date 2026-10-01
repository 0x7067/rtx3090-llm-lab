#!/usr/bin/env bash
set -euo pipefail

task_dir=$(cd "$(dirname "$0")" && pwd)
harness=$task_dir/../paired-harness/harness.py
plan=$task_dir/speed-plans/plan-speed.json
image=local/strata-baseline:30ec18e
export KUBECONFIG=/home/denguinho/.kube/config
export UV_CACHE_DIR=/tmp/uv-cache
read -r -a paused_pids <<< "${PAUSED_PIDS:-}"
original_replicas=$(kubectl -n apps get deployment llama -o jsonpath='{.spec.replicas}')
original_suspend=$(kubectl -n flux-system get kustomization apps -o jsonpath='{.spec.suspend}')
original_suspend=${original_suspend:-false}
active_container=

restore() {
  local status=$?
  trap - EXIT INT TERM
  [[ -z $active_container ]] || docker rm -f "$active_container" >/dev/null 2>&1 || true
  kubectl -n apps scale deployment llama --replicas="$original_replicas" >/dev/null 2>&1 || status=1
  if [[ $original_suspend != true ]]; then
    flux resume kustomization apps >/dev/null 2>&1 || status=1
    flux reconcile kustomization apps --with-source >/dev/null 2>&1 || status=1
  fi
  kubectl -n apps rollout status deployment/llama --timeout=20m || status=1
  if ((${#paused_pids[@]})); then
    kill -CONT "${paused_pids[@]}" 2>/dev/null || true
  fi
  exit "$status"
}
trap restore EXIT INT TERM

wait_ready() {
  for _ in $(seq 1 180); do
    curl -fsS http://127.0.0.1:18182/health >/dev/null && return 0
    [[ $(docker inspect "$active_container" --format '{{.State.Running}}') == true ]] || return 1
    sleep 5
  done
  return 1
}

run_variant() {
  local label=$1
  local config_dir=$task_dir/strata-speed-configs/$label
  local prefix=$task_dir/strata-speed-$label
  for path in "$prefix-run" "$prefix-summary.json" "$prefix-summary.md" "$prefix-server.log" "$prefix-harness.log" "$prefix-telemetry.tsv" "$prefix-gpu-telemetry.tsv"; do
    [[ ! -e $path ]] || { echo "Refusing to overwrite $path" >&2; return 1; }
  done
  active_container=strata-speed-$label
  docker run -d --name "$active_container" --gpus all --network host \
    --memory 44g --memory-swap 44g --cpus 8 --ulimit memlock=-1:-1 \
    -v /data/strata-baseline:/data \
    -v "$config_dir:/data/config" \
    -e VISION=no -e HOST=127.0.0.1 -e PORT=18182 -e FAMILY=qwen \
    -e MODEL=IQ2_XS -e CONTEXT=150000 -e KV=int8 -e LOW_RAM=on -e GPU=0 \
    "$image" >/dev/null
  bash "$task_dir/monitor.sh" "$active_container" "$prefix-telemetry.tsv" &
  local monitor_pid=$!
  (
    while [[ $(docker inspect "$active_container" --format '{{.State.Running}}' 2>/dev/null || true) == true ]]; do
      observed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      guard=$(systemctl is-active vram-thermal-guard.service 2>/dev/null || true)
      [[ -e /run/gpu-thermal-trip ]] && trip=true || trip=false
      gpu=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit,memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits 2>/dev/null || true)
      printf '%s\tguard=%s\ttrip=%s\t%s\n' "$observed_at" "$guard" "$trip" "$gpu" >> "$prefix-gpu-telemetry.tsv"
      sleep 5
    done
  ) &
  local gpu_pid=$!
  wait_ready
  uv run --python /usr/bin/python3 --no-project python "$harness" run \
    --plan "$plan" --arm strata_flash_next --out "$prefix-run" --timeout 1800 \
    > "$prefix-harness.log" 2>&1
  docker logs "$active_container" > "$prefix-server.log" 2>&1
  docker stop -t 30 "$active_container" >/dev/null
  wait "$monitor_pid" || true
  wait "$gpu_pid" || true
  ! grep -q '^abort' "$prefix-telemetry.tsv"
  uv run --python /usr/bin/python3 --no-project python "$harness" summarize \
    --plan "$plan" --run "$prefix-run" --out "$prefix-summary.json" --markdown "$prefix-summary.md"
  docker rm "$active_container" >/dev/null
  active_container=
}

[[ $(docker image inspect "$image" --format '{{.Id}}') == sha256:f0aead168c875cb4035ad7e8647c015939658866e7f37089bb2a126f2f06dda4 ]]
[[ ! -e /run/gpu-thermal-trip ]]
systemctl is-active --quiet vram-thermal-guard.service
systemctl is-active --quiet vram-fan-curve.service
if [[ $original_suspend != true ]]; then flux suspend kustomization apps >/dev/null; fi
kubectl -n apps scale deployment llama --replicas=0 >/dev/null
kubectl -n apps wait --for=delete pod -l app.kubernetes.io/name=llama --timeout=10m
for _ in $(seq 1 120); do
  [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] && break
  sleep 5
done
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
for variant in ${VARIANTS:-control calibrated}; do
  run_variant "$variant"
done
