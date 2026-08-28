#!/usr/bin/env bash
# Suspend Flux reconciliation, scale down the qwen3.8 (llama) deployment,
# wait for the GPU to free, then start Tiel under llama.cpp on port 8090.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_FILE=models/Tiel-Coder-35B-A3B-UD-Q4_K_S.gguf
CTX="${CTX:-65536}"

echo "== suspending flux kustomization 'apps'"
flux suspend kustomization apps

echo "== scaling down apps/llama"
kubectl -n apps scale deploy llama --replicas=0
kubectl -n apps wait --for=delete pod -l app.kubernetes.io/name=llama --timeout=180s || true

echo "== waiting for VRAM to free"
for i in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$u" -lt 500 ] && break
  sleep 3
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "== starting Tiel llama-server on :8090"
docker run -d --name tiel --gpus all \
  -p 127.0.0.1:8090:8080 \
  -v "$PWD/models:/models:ro" \
  ghcr.io/ggml-org/llama.cpp:full-cuda \
  --server -m "/models/$(basename "$MODEL_FILE")" \
  -ngl 99 -c "$CTX" -np 4 --jinja \
  --host 0.0.0.0 --port 8080

echo "== waiting for model load"
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 2 http://127.0.0.1:8090/health || true)
  [ "$code" = "200" ] && { echo "Tiel ready"; exit 0; }
  sleep 5
done
echo "Tiel did not become ready; last container logs:" >&2
docker logs --tail 30 tiel >&2
exit 1
