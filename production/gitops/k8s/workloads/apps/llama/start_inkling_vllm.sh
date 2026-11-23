#!/usr/bin/env bash
# llama-swap owns this worker and unloads it before starting another GPU model.
set -Eeuo pipefail

runtime=/opt/inkling-vllm
model=/models/inkling-vllm
port=${1:?expected the worker port}
served_model_name=${2:-inkling-2-mini}
export PYTHONHOME="$runtime/usr"
export PYTHONPATH="/opt/inkling-sm86-opt:$runtime/usr/local/lib/python3.14/dist-packages"
export PYTHONNOUSERSITE=1
export CUDA_HOME="$runtime/usr/local/cuda-13.2"
export CPATH="$runtime/usr/include"
export PATH="$runtime/usr/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:$CUDA_HOME/lib64"
export XDG_CACHE_HOME=/cache/inkling-vllm
export HOME="$XDG_CACHE_HOME"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export INKLING_SM86_LMHEAD_INT8="${INKLING_SM86_LMHEAD_INT8:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p "$XDG_CACHE_HOME"

# The media allowlist contains a path delimiter, which cannot occur in a hostname.
exec "$runtime/usr/bin/python3.14" -m vllm.entrypoints.cli.main serve "$model" \
  --host 127.0.0.1 --port "$port" --served-model-name "$served_model_name" --dtype bfloat16 \
  --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 512 \
  --long-prefill-token-threshold 256 --long-prefill-token-threshold-adaptive \
  --cache-aware-admission-window 8 \
  --gpu-memory-utilization 0.98 --kv-cache-memory-bytes 5959267123 --kv-cache-dtype fp8 \
  --limit-mm-per-prompt.image=1 --allowed-media-domains 'inline-only/' \
  --cpu-offload-gb 2 --cpu-offload-params embed_tokens visual markov_w1 \
  --compilation-config.cudagraph_mode=FULL_DECODE_ONLY \
  --compilation-config.decode_cudagraph_capture_request_sizes='[1,2,4,8]' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling \
  --enable-prompt-tokens-details
