#!/usr/bin/env bash
# Run the qwen3.8 vLLM profile standalone on :8094 for the MMLU-Pro comparison.
# Mirrors the k8s deployment that git commit 0a3a258 replaced: same image, same
# env, same host paths. The Flux-managed llama Deployment is only scaled to 0,
# never edited, so restoring is a scale back to 1.
set -euo pipefail
# Path to the llama workload directory in your cluster manifests repo; only
# the chat_template.jinja beside the Deployment is read.
REPO=${CLUSTER_REPO:-../cluster}/k8s/workloads/apps/llama

docker rm -f qwenbench >/dev/null 2>&1 || true
docker run -d --name qwenbench --gpus all --user 1000:1000 \
  -p 127.0.0.1:8094:8080 \
  --shm-size 27g \
  -v ${MODELS_DIR:-$PWD/models}:/app/models \
  -v ${CACHE_DIR:-$PWD/cache}:/cache \
  -v "$REPO/chat_template.jinja:/app/config/chat_template.jinja:ro" \
  -e TZ=America/Sao_Paulo -e PORT=8080 -e CTX=long -e VISION=1 \
  -e MAX_LEN=140000 -e GPU_UTIL=0.95 -e PREFIX_CACHE=1 \
  -e DRAFT_TOKENS=3 -e ASYNC_SCHED=0 -e VERIFY=0 -e HOME=/cache \
  -e 'EXTRA_ARGS=--kv-offloading-size 24 --kv-transfer-config {"kv_connector_extra_config":{"offload_prompt_only":false}} --enable-cumem-allocator --chat-template /app/config/chat_template.jinja --enable-auto-tool-choice --tool-call-parser qwen3_coder' \
  qwen38-27b-3090:v7 single >/dev/null

# vLLM cold start can take many minutes (JIT + torch.compile), matching the
# deployment's failureThreshold of 120 x 10s.
for i in $(seq 1 180); do
  c=$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:8094/health || true)
  [ "$c" = "200" ] && { echo "QWEN READY after $((i*10))s"; exit 0; }
  s=$(docker inspect -f '{{.State.Running}}' qwenbench 2>/dev/null || echo false)
  [ "$s" = "false" ] && { echo "QWEN DIED"; docker logs --tail 25 qwenbench; exit 1; }
  sleep 10
done
echo "QWEN TIMEOUT"; docker logs --tail 25 qwenbench; exit 1
