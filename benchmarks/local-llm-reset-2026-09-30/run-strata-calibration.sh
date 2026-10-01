#!/usr/bin/env bash
set -euo pipefail

task_dir=$(cd "$(dirname "$0")" && pwd)
container=strata-calibration
image=local/strata-baseline:30ec18e
result=$task_dir/strata-publisher-calibration.log
engine_log=$task_dir/strata-publisher-calibration-engine.log
telemetry=$task_dir/strata-publisher-calibration-telemetry.tsv
gpu_telemetry=$task_dir/strata-publisher-calibration-gpu-telemetry.tsv
export KUBECONFIG=/home/denguinho/.kube/config
read -r -a paused_pids <<< "${PAUSED_PIDS:-}"
original_replicas=$(kubectl -n apps get deployment llama -o jsonpath='{.spec.replicas}')
original_suspend=$(kubectl -n flux-system get kustomization apps -o jsonpath='{.spec.suspend}')
original_suspend=${original_suspend:-false}

restore() {
  local status=$?
  trap - EXIT INT TERM
  docker rm -f "$container" >/dev/null 2>&1 || true
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

for path in "$result" "$engine_log" "$telemetry" "$gpu_telemetry"; do
  [[ ! -e $path ]] || { echo "Refusing to overwrite $path" >&2; exit 1; }
done
[[ $(docker image inspect "$image" --format '{{.Id}}') == sha256:f0aead168c875cb4035ad7e8647c015939658866e7f37089bb2a126f2f06dda4 ]]
[[ ! -e /run/gpu-thermal-trip ]]
systemctl is-active --quiet vram-thermal-guard.service
systemctl is-active --quiet vram-fan-curve.service

if [[ $original_suspend != true ]]; then
  flux suspend kustomization apps >/dev/null
fi
kubectl -n apps scale deployment llama --replicas=0 >/dev/null
kubectl -n apps wait --for=delete pod -l app.kubernetes.io/name=llama --timeout=10m
for _ in $(seq 1 120); do
  [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] && break
  sleep 5
done
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]

docker run -d --name "$container" --gpus all --network host \
  --memory 44g --memory-swap 44g --cpus 8 --ulimit memlock=-1:-1 \
  -v /data/strata-baseline:/data \
  --entrypoint sh "$image" -lc \
  'cd /opt/strata && .venv/bin/python tools/calibrate.py /data/config/strata-iq2_xs.json' >/dev/null
bash "$task_dir/monitor.sh" "$container" "$telemetry" &
monitor_pid=$!
(
  while [[ $(docker inspect "$container" --format '{{.State.Running}}' 2>/dev/null || true) == true ]]; do
    observed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    guard=$(systemctl is-active vram-thermal-guard.service 2>/dev/null || true)
    [[ -e /run/gpu-thermal-trip ]] && trip=true || trip=false
    gpu=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit,memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits 2>/dev/null || true)
    printf '%s\tguard=%s\ttrip=%s\t%s\n' "$observed_at" "$guard" "$trip" "$gpu" >> "$gpu_telemetry"
    sleep 5
  done
) &
gpu_pid=$!
docker logs -f "$container" | tee "$result"
status=$(docker inspect "$container" --format '{{.State.ExitCode}}')
docker cp "$container:/opt/strata/strata-iq2_xs.log" "$engine_log" 2>/dev/null || true
wait "$monitor_pid" || true
wait "$gpu_pid" || true
[[ $status == 0 ]]
[[ ! -e /run/gpu-thermal-trip ]]
! grep -q '^abort' "$telemetry"
