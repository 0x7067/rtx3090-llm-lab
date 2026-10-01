#!/usr/bin/env bash
set -euo pipefail

task_dir=$(cd "$(dirname "$0")" && pwd)
harness=$task_dir/../paired-harness/harness.py
plan=$task_dir/speed-plans/plan-speed.json
image=local/llm-baseline-vllm:0.30.0
model=/data/models/qwen3.8/orcarouter/OrcaSAQ2-27B
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
  local server_log=$1
  for _ in $(seq 1 240); do
    curl -fsS http://127.0.0.1:18180/health >/dev/null && return 0
    if [[ $(docker inspect "$active_container" --format '{{.State.Running}}') != true ]]; then
      docker logs "$active_container" > "$server_log" 2>&1
      return 1
    fi
    sleep 5
  done
  return 1
}

run_variant() {
  local label=$1
  local depth=$2
  local batch=$3
  local prefix=$task_dir/orca-speed-$label
  for path in "$prefix-run" "$prefix-summary.json" "$prefix-summary.md" "$prefix-server.log" "$prefix-harness.log" "$prefix-metrics.txt" "$prefix-telemetry.tsv" "$prefix-gpu-telemetry.tsv"; do
    [[ ! -e $path ]] || { echo "Refusing to overwrite $path" >&2; return 1; }
  done
  active_container=orca-speed-$label
  local extra
  extra="--speculative-config {\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":$depth}"
  docker run -d --name "$active_container" --gpus all --network host --ipc host \
    --memory 44g --memory-swap 44g --cpus 8 \
    -v "$model:/models/orca:ro" \
    -e PRESET=16gb-mtp -e MODEL=/models/orca -e SERVED_NAME=orca-qwen3.8-27b \
    -e PORT=18180 -e BIND=127.0.0.1 -e MAXLEN=150000 -e MAXSEQS=8 \
    -e MAXBATCHTOK="$batch" -e EXTRA="$extra" \
    --entrypoint /opt/orcasaq2/presets/serve.sh "$image" >/dev/null
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
  wait_ready "$prefix-server.log"
  uv run --python /usr/bin/python3 --no-project python "$harness" run \
    --plan "$plan" --arm orca_vllm --out "$prefix-run" --timeout 1800 \
    > "$prefix-harness.log" 2>&1
  curl -fsS http://127.0.0.1:18180/metrics > "$prefix-metrics.txt"
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

pick_depth() {
  uv run --python /usr/bin/python3 --no-project python - "$task_dir" <<'PY'
import json
import math
import sys
from pathlib import Path
root = Path(sys.argv[1])
scores = {}
for depth in (1, 2, 3):
    data = json.loads((root / f"orca-speed-mtp{depth}-mbt2048-summary.json").read_text())
    perf = data["performance"]
    values = [perf[t]["metrics"]["e2e_output_tps"]["median"] for t in ("decode-short", "decode-medium")]
    scores[depth] = math.prod(values) ** (1 / len(values))
print(max(scores, key=scores.get))
PY
}

[[ $(docker image inspect "$image" --format '{{.Id}}') == sha256:1d60e0a2fc9eb37042f94b8d5fb23457df6a27ca365305854b1bbcea8b4d9308 ]]
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
if [[ -n ${ONLY_LABEL:-} ]]; then
  run_variant "$ONLY_LABEL" "$ONLY_DEPTH" "$ONLY_BATCH"
  exit 0
fi
for depth in 1 2 3; do
  run_variant "mtp${depth}-mbt2048" "$depth" 2048
done
winner=$(pick_depth)
printf '%s\n' "$winner" > "$task_dir/orca-speed-depth-winner.txt"
run_variant "mtp${winner}-mbt8192" "$winner" 8192
