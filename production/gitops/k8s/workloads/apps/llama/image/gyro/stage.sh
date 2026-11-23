#!/usr/bin/env bash
# Run the frozen profile with an independent cleanup path and bounded resources.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-/tmp/gyro-release/staging}
mkdir -p "$out"
test "$(kubectl -n flux-system get kustomization apps -o jsonpath='{.spec.suspend}')" != true
cleanup() {
  result=$?
  set +e
  trap - EXIT INT TERM
  docker logs gyro-m-staging > "$out/server.log" 2>&1
  docker cp gyro-m-staging:/tmp/gyro-m-engine.log "$out/engine.log" 2>/dev/null
  docker stop -t 30 gyro-m-staging >/dev/null 2>&1 || true
  docker rm gyro-m-staging >/dev/null 2>&1 || true
  if [[ -n ${monitor_pid:-} ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
  kubectl -n apps scale deployment llama --replicas=1
  kubectl -n flux-system patch kustomization apps --field-manager=flux-client-side-apply --type=merge -p '{"spec":{"suspend":false}}'
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
kubectl -n flux-system patch kustomization apps --field-manager=flux-client-side-apply --type=merge -p '{"spec":{"suspend":true}}'
pod=$(kubectl -n apps get pods -l app.kubernetes.io/name=llama --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
kubectl -n apps scale deployment llama --replicas=0
kubectl -n apps wait --for=delete "pod/$pod" --timeout=180s
docker run -d --name gyro-m-staging --gpus all --network host \
  --user 1000:1000 --memory 40g --memory-swap 40g --cpus 8 \
  --ulimit memlock=-1:-1 --stop-timeout 30 \
  -v /data/gyro-m:/models/gyro-m:ro \
  -v /data/gyro-m/config:/config:ro \
  -e STRATA_RESIDENT_HEADROOM_GIB=8 \
  --entrypoint /opt/strata/.venv/bin/python home-llm-gyro:20261016-v1 \
  /opt/gyro/serve/server.py --engine strata --config /config/gyro-m.json \
  --host 127.0.0.1 --port 18183
(
  while [[ $(docker inspect gyro-m-staging --format '{{.State.Running}}') == true ]]; do
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    pressure=$(awk '$1 == "some" {split($2,v,"="); print v[2]}' /proc/pressure/memory)
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$available" "$pressure" \
      "$(nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits)" \
      "$(docker stats --no-stream --format '{{.MemUsage}}' gyro-m-staging)" >> "$out/telemetry.tsv"
    if [[ $available -lt 8388608 ]] || \
        awk -v p="$pressure" 'BEGIN {exit !(p > 10)}'; then
      echo 'Host pressure: aborting staging' >> "$out/telemetry.tsv"
      docker stop -t 20 gyro-m-staging
      exit 1
    fi
    sleep 5
  done
) &
monitor_pid=$!
ready=false
for ((i=0; i<180; i++)); do
  if curl -fsS http://127.0.0.1:18183/health >/dev/null 2>&1; then ready=true; break; fi
  [[ $(docker inspect gyro-m-staging --format '{{.State.Running}}') == true ]] || break
  sleep 5
done
docker logs gyro-m-staging > "$out/startup.log" 2>&1
[[ $ready == true ]] || { echo 'Strata did not become ready'; exit 1; }
UV_CACHE_DIR=/tmp/gyro-uv uv run --no-project python "$here/qualify.py" \
  --base http://127.0.0.1:18183/v1 --out "$out" --long
docker cp gyro-m-staging:/tmp/gyro-m-engine.log "$out/engine.log"
docker logs gyro-m-staging > "$out/server.log" 2>&1
