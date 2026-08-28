#!/usr/bin/env bash
# The one window where the GPU is not serving production.
#
# Runs, in order: the qwen3.8 half of the quality comparison, then the Tiel
# parallelization sweep, then restores the deployment. Both need the whole card,
# so they share a single outage instead of two.
#
# Ordering that matters: Flux reconciles apps/llama at replicas 1, so suspend
# BEFORE scaling to 0 or the next reconcile puts Tiel back on the card next to
# whatever this script started. restore_qwen.sh resumes Flux at the end, and the
# trap runs it even if a step fails, so the card never stays parked.
#
# Usage: bench_window.sh   (expect roughly two hours)
set -uo pipefail
cd "$(dirname "$0")"
QWEN=http://127.0.0.1:8094

restore() {
  echo "== restoring deployment"
  ./restore_qwen.sh || echo "RESTORE FAILED - check kubectl -n apps get pods" >&2
}
trap restore EXIT

echo "== suspending flux (must precede the scale-down)"
flux suspend kustomization apps

echo "== scaling apps/llama to 0 and waiting for VRAM"
kubectl -n apps scale deploy llama --replicas=0
kubectl -n apps wait --for=delete pod -l app.kubernetes.io/name=llama --timeout=300s || true
for _ in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$u" -lt 500 ] && break
  sleep 3
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "== starting qwen3.8 (vLLM cold start takes several minutes)"
./run_qwen.sh

echo "== qwen HumanEval"
uv run bench_quality.py "$QWEN" qwen3.8-27b HumanEval.jsonl cand_qwen_v2.jsonl 6
rm -rf work_qwen_v2 && mkdir -p work_qwen_v2
cp cand_qwen_v2.jsonl work_qwen_v2/candidates.jsonl
cp sandbox_runner.py work_qwen_v2/
docker run --rm --network none --memory 4g --cpus 4 \
  -v "$PWD/work_qwen_v2:/work" python:3.12-slim python /work/sandbox_runner.py
echo "QWEN_HUMANEVAL_DONE"

echo "== qwen multi-turn (same mutants.jsonl, byte-identical to Tiel's)"
uv run bench_multiturn.py "$QWEN" qwen3.8-27b mutants.jsonl \
  results_multiturn_qwen_v2.json work_mt_qwen_v2 6
echo "QWEN_MULTITURN_DONE"

echo "== qwen speed, matched at 4 concurrent"
uv run bench_speed.py "$QWEN" qwen3.8-27b results_speed_qwen_v2.json || \
  echo "speed run failed, not fatal"

echo "== freeing the card for the Tiel sweep"
docker rm -f qwenbench >/dev/null 2>&1 || true
for _ in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$u" -lt 500 ] && break
  sleep 3
done

echo "== Tiel parallelization sweep"
./sweep_parallel.sh results_parallel.json
echo "SWEEP_DONE"

echo "ALL_DONE"
