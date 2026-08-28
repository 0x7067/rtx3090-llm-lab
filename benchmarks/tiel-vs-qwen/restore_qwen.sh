#!/usr/bin/env bash
# Give the GPU back to the production deployment after a benchmark window.
#
# The name is historical. When this was written the Deployment served qwen3.8,
# so restoring it meant restoring qwen. Since 2026-08-25 the Deployment serves
# Tiel, and this script restores that; the benchmark container it tears down is
# the standalone qwen profile started by /tmp/run_qwen.sh.
#
# Pair with the teardown, which must suspend Flux BEFORE scaling. Flux
# reconciles apps/llama at replicas: 1, so an unsuspended scale-to-0 is undone
# on the next reconcile and both models fight for the card.
set -euo pipefail

echo "== removing benchmark containers"
docker rm -f qwenbench oblitbench sweep tt sp tielv tiel 2>/dev/null || true

# Container removal can finish before the GPU driver releases its VRAM.
echo "== waiting for VRAM to free"
for _ in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$u" -lt 500 ] && break
  sleep 3
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
[ "$u" -ge 500 ] && echo "WARNING: VRAM did not drop below 500 MiB; continuing production restore" >&2

echo "== scaling up apps/llama and resuming flux"
kubectl -n apps scale deploy llama --replicas=1
flux resume kustomization apps --timeout 5m || true

echo "== waiting for health"
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 2 http://llama.apps.svc.cluster.local:8080/health || true)
  [ "$code" = "200" ] && { echo "deployment restored"; exit 0; }
  sleep 5
done
echo "not healthy after 10m; check: kubectl -n apps get pods" >&2
exit 1
