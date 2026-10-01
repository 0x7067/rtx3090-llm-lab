#!/usr/bin/env bash
set -euo pipefail

task_dir=$(cd "$(dirname "$0")" && pwd)
harness=$task_dir/../paired-harness/harness.py
plans=$task_dir/followup-plans
strata_image=local/strata-baseline:30ec18e
strata_digest=sha256:f0aead168c875cb4035ad7e8647c015939658866e7f37089bb2a126f2f06dda4
export KUBECONFIG=/home/denguinho/.kube/config
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PYTHON_INSTALL_DIR=/tmp/uv-python
export PATH=/home/denguinho/.local/bin:/usr/local/bin:/usr/bin:/bin

original_replicas=$(kubectl -n apps get deployment llama -o jsonpath='{.spec.replicas}')
original_suspend=$(kubectl -n flux-system get kustomization apps -o jsonpath='{.spec.suspend}')
original_suspend=${original_suspend:-false}
active_container=
read -r -a paused_pids <<< "${PAUSED_PIDS:-}"

restore() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n $active_container ]]; then
    docker rm -f "$active_container" >/dev/null 2>&1 || true
  fi
  local restored=false
  local attempt
  for ((attempt = 1; attempt <= 60; attempt++)); do
    if kubectl get --raw=/readyz >/dev/null 2>&1; then
      kubectl -n apps scale deployment llama --replicas="$original_replicas" >/dev/null 2>&1 || true
      if [[ $original_suspend != true ]]; then
        flux resume kustomization apps >/dev/null 2>&1 || true
        flux reconcile kustomization apps --with-source >/dev/null 2>&1 || true
      fi
      if [[ $(kubectl -n apps get deployment llama -o jsonpath='{.spec.replicas}' 2>/dev/null) == "$original_replicas" ]] &&
         [[ $original_suspend == true || $(kubectl -n flux-system get kustomization apps -o jsonpath='{.spec.suspend}' 2>/dev/null) != true ]]; then
        restored=true
        break
      fi
    fi
    sleep 5
  done
  if [[ $restored == true ]]; then
    kubectl -n apps rollout status deployment/llama --timeout=20m || status=1
  else
    status=1
  fi
  if ((${#paused_pids[@]})); then
    kill -CONT "${paused_pids[@]}" 2>/dev/null || true
  fi
  exit "$status"
}

trap restore EXIT INT TERM

wait_ready() {
  local container=$1
  local attempts=$2
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if curl -fsS http://127.0.0.1:18182/health >/dev/null; then
      return 0
    fi
    if [[ $(docker inspect "$container" --format '{{.State.Running}}') != true ]]; then
      docker logs "$container"
      return 1
    fi
    sleep 5
  done
  docker logs --tail 200 "$container"
  return 1
}

thermal_log() {
  local container=$1
  local path=$2
  while [[ $(docker inspect "$container" --format '{{.State.Running}}' 2>/dev/null || true) == true ]]; do
    local observed_at guard trip gpu
    observed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    guard=$(systemctl is-active vram-thermal-guard.service 2>/dev/null || true)
    [[ -e /run/gpu-thermal-trip ]] && trip=true || trip=false
    gpu=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit,memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits 2>/dev/null || true)
    printf '%s\tguard=%s\ttrip=%s\t%s\n' "$observed_at" "$guard" "$trip" "$gpu" >> "$path"
    sleep 5
  done
}

run_plan() {
  local label=$1
  local plan=$2
  local run_dir=$task_dir/strata-$label-run
  local summary=$task_dir/strata-$label-summary.json
  local summary_md=$task_dir/strata-$label-summary.md
  local server_log=$task_dir/strata-$label-server.log
  local harness_log=$task_dir/strata-$label-harness.log
  local telemetry=$task_dir/strata-$label-telemetry.tsv
  local thermal=$task_dir/strata-$label-gpu-telemetry.tsv

  for path in "$run_dir" "$summary" "$summary_md" "$server_log" "$harness_log" "$telemetry" "$thermal"; do
    if [[ -e $path ]]; then
      echo "Refusing to overwrite $path" >&2
      return 1
    fi
  done

  active_container=strata-$label
  docker run -d --name "$active_container" --gpus all --network host \
    --memory 44g --memory-swap 44g --cpus 8 --ulimit memlock=-1:-1 \
    -v /data/strata-baseline:/data \
    -e VISION=no -e HOST=127.0.0.1 -e PORT=18182 -e FAMILY=qwen \
    -e MODEL=IQ2_XS -e CONTEXT=150000 -e KV=int8 -e LOW_RAM=on -e GPU=0 \
    "$strata_image" >/dev/null

  bash "$task_dir/monitor.sh" "$active_container" "$telemetry" > "$task_dir/strata-$label-monitor.log" 2>&1 &
  local monitor_pid=$!
  thermal_log "$active_container" "$thermal" &
  local thermal_pid=$!

  wait_ready "$active_container" 180
  uv run --python /usr/bin/python3 --no-project python "$harness" run \
    --plan "$plan" --arm strata_flash_next --out "$run_dir" --timeout 1800 \
    > "$harness_log" 2>&1

  docker logs "$active_container" > "$server_log" 2>&1
  docker stop -t 30 "$active_container" >/dev/null
  wait "$monitor_pid"
  wait "$thermal_pid"
  if grep -q '^abort' "$telemetry"; then
    return 1
  fi

  uv run --python /usr/bin/python3 --no-project python "$harness" summarize \
    --plan "$plan" --run "$run_dir" --out "$summary" --markdown "$summary_md"
  docker rm "$active_container" >/dev/null
  active_container=
}

[[ $(docker image inspect "$strata_image" --format '{{.Id}}') == "$strata_digest" ]]
[[ ! -e /run/gpu-thermal-trip ]]
systemctl is-active --quiet vram-thermal-guard.service
systemctl is-active --quiet vram-fan-curve.service
offsets=$(/opt/gpu-tune/venv/bin/python -c 'import pynvml as n; n.nvmlInit(); d=n.nvmlDeviceGetHandleByIndex(0); print(n.nvmlDeviceGetGpcClkVfOffset(d), n.nvmlDeviceGetMemClkVfOffset(d))')
[[ $offsets == "100 0" ]]

for ((attempt = 1; attempt <= 120; attempt++)); do
  memory_pressure=$(awk '$1 == "some" {split($2, value, "="); print value[2]}' /proc/pressure/memory)
  io_pressure=$(awk '$1 == "some" {split($2, value, "="); print value[2]}' /proc/pressure/io)
  if awk -v memory="$memory_pressure" -v io="$io_pressure" 'BEGIN {exit !(memory <= 1 && io <= 20)}'; then
    break
  fi
  sleep 5
done
awk -v memory="$memory_pressure" -v io="$io_pressure" 'BEGIN {exit !(memory <= 1 && io <= 20)}'

if [[ $original_suspend != true ]]; then
  flux suspend kustomization apps >/dev/null
fi
kubectl -n apps scale deployment llama --replicas=0 >/dev/null
kubectl -n apps wait --for=delete pod -l app.kubernetes.io/name=llama --timeout=10m
for _ in $(seq 1 120); do
  if [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
    break
  fi
  sleep 5
done
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]

if [[ ${SKIP_FULL:-0} != 1 ]]; then
  run_plan full-32768 "$plans/plan-full-32768.json"
fi
for policy in ${POLICIES:-control nonthinking-simple nonthinking-coding}; do
  run_plan "followup-$policy" "$plans/plan-$policy.json"
done
