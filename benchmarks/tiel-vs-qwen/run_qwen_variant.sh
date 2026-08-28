#!/usr/bin/env bash
# Start the standalone qwen3.8 vLLM container on :8094 with variant env for the
# overnight sweeps. Same image/volumes as run_qwen.sh; the deployment must
# already be scaled to 0 (see bench_window.sh for the flux/scale dance).
#
# Overridable: IMAGE (qwen38-27b-3090:v8), DRAFT_TOKENS (3), ASYNC_SCHED (0), MAX_LEN (140000),
# GPU_UTIL (0.95), DRAFT_ATTN_BACKEND, DRAFT_KV_DTYPE, DRAFT_SAMPLE,
# APPEND_ARGS (space-free tokens appended after the stock EXTRA_ARGS, so a
# repeated vLLM flag overrides the stock one — FlexibleArgumentParser keeps
# the last occurrence).
set -euo pipefail
REPO=/data/development/projects/docker-services/k8s/workloads/apps/llama

BASE_EXTRA='--kv-offloading-size 24 --kv-transfer-config {"kv_connector_extra_config":{"offload_prompt_only":false}} --enable-cumem-allocator --chat-template /app/config/chat_template.jinja --enable-auto-tool-choice --tool-call-parser qwen3_coder --limit-mm-per-prompt {"image":{"count":1}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":2097152}}'

docker rm -f qwenbench >/dev/null 2>&1 || true
docker run -d --name qwenbench --gpus all --user 1000:1000 \
  -p 127.0.0.1:8094:8080 \
  --shm-size 27g \
  -v /data/buttercup_6tb/k3s/vllm-trial/models:/app/models \
  -v /data/buttercup_6tb/k3s/vllm-trial/cache:/cache \
  -v "$REPO/chat_template.jinja:/app/config/chat_template.jinja:ro" \
  -e TZ=America/Sao_Paulo -e PORT=8080 -e CTX=long -e VISION=1 \
  -e MAX_LEN="${MAX_LEN:-140000}" -e GPU_UTIL="${GPU_UTIL:-0.95}" -e PREFIX_CACHE=1 \
  -e DRAFT_TOKENS="${DRAFT_TOKENS:-3}" -e ASYNC_SCHED="${ASYNC_SCHED:-0}" \
  ${DRAFT_ATTN_BACKEND:+-e DRAFT_ATTN_BACKEND="$DRAFT_ATTN_BACKEND"} \
  ${DRAFT_KV_DTYPE:+-e DRAFT_KV_DTYPE="$DRAFT_KV_DTYPE"} \
  ${DRAFT_SAMPLE:+-e DRAFT_SAMPLE="$DRAFT_SAMPLE"} \
  -e VERIFY=0 -e HOME=/cache \
  -e "EXTRA_ARGS=${BASE_EXTRA}${APPEND_ARGS:+ $APPEND_ARGS}" \
  "${IMAGE:-qwen38-27b-3090:v8}" single >/dev/null

for i in $(seq 1 180); do
  c=$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:8094/health || true)
  [ "$c" = "200" ] && { echo "QWEN READY after $((i*10))s"; exit 0; }
  s=$(docker inspect -f '{{.State.Running}}' qwenbench 2>/dev/null || echo false)
  [ "$s" = "false" ] && { echo "QWEN DIED"; docker logs --tail 40 qwenbench; exit 1; }
  sleep 10
done
echo "QWEN TIMEOUT"; docker logs --tail 40 qwenbench; exit 1
