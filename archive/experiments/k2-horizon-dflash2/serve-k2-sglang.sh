#!/usr/bin/env bash
# Serve K2-Horizon-7B with SGLang for data regeneration (bf16, Ampere).
# Short sequences + batching: this is a throughput job, not a latency one.
set -euo pipefail
W=/data/buttercup_6tb/specforge-work
export HF_HOME="$W/hf-home" UV_CACHE_DIR="$W/.uv-cache"
export CUDA_HOME="$W/venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK ${NVCC_PREPEND_FLAGS:-}"
cd "$W"; source venv/bin/activate
exec python -m sglang.launch_server \
  --model-path "$W/models/IFM/K2-Horizon-7B" \
  --trust-remote-code \
  --dtype bfloat16 \
  --attention-backend triton \
  --context-length "${CTX:-16384}" \
  --mem-fraction-static "${MEM_FRACTION:-0.90}" \
  --max-running-requests "${MAX_RUNNING:-16}" \
  --reasoning-parser k2_horizon \
  --tool-call-parser k2_horizon \
  --host 127.0.0.1 --port "${PORT:-30000}"
