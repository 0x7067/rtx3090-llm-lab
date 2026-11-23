#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]{1,5}$ ]] || ((10#$1 < 1 || 10#$1 > 65535)); then
    echo 'Usage: serve-long.sh PORT' >&2
    exit 2
fi

export PYTHONPATH=/opt/orca/python:/opt/orca/plugin:/opt/orca/gguf-source
export LD_LIBRARY_PATH=/native:/native-cuda:/usr/local/cuda/lib64
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/cache/orca HF_HOME=/cache/orca/huggingface VLLM_CACHE_ROOT=/cache/orca/vllm
export VLLM_NO_USAGE_STATS=1 TORCHINDUCTOR_COMPILE_THREADS=1 VLLM_COMPUTE_NANS_IN_LOGITS=1
export SPARSE_KV=1 SPARSE_IMPL=triton SPARSE_BASIS=/opt/sparsekv/venv/share/sparsekv/basis27.pt
export SPARSEKV_ROOT=/opt/sparsekv/venv/lib/python3.12/site-packages
export SPARSE_KSEL=2048 SPARSE_SINK=64 SPARSE_WIN=1024 SPARSE_MEM_RESERVE=1 SPARSE_SWITCH=1
# The selection sketch is separate from the packed Q8 key and value cache.
export SPARSE_SKETCH_BITS=4 SPARSE_STATS_SUB=1 SPARSE_SELECT_EVERY=1
export SPARSE_TIER=1 SPARSE_TIER_SPILL=1 SPARSE_TIER_SPILL_POLICY=eager SPARSE_TIER_HOT_FRAC=0.025
export SPARSE_TIER_HOST_BITS=8 SPARSE_TIER_SUFFIX_SPARSE=0 SPARSE_TIER_STAGE_SERIAL=1
export SPARSE_TIER_HOST_PREFIX=1 SPARSE_TIER_HOST_MAMBA_SLOTS=8 SPARSE_TIER_HOST_MAMBA_PER_REQ=2
export SPARSE_TIER_MAMBA_SAVE_AT_FINISH=1 SPARSE_TIER_PIN_PREFIX_MAX=2
export SPARSE_PREFILL_YIELD_SHORT_BYPASS=2600 SPARSE_PREFILL_SHARE_CHUNK=256
export PTHFA_PREFILL_FA2=1 VLLM_VISION_CPU_OFFLOAD_GB=1

context=163840
slots=4
export SPARSE_TIER_HOST_TOKENS=$((context * slots))

exec /opt/sparsekv/venv/bin/vllm serve /opt/orca/weights/model.gguf \
    --hf-config-path /opt/orca/model --tokenizer /opt/orca/model \
    --chat-template /opt/orca/chat_template.jinja --served-model-name orcasaq2-cyber \
    --host 127.0.0.1 --port "$1" --attention-backend CUSTOM \
    --kv-cache-dtype int8_per_token_head --max-model-len "$context" --max-num-seqs "$slots" \
    --max-num-batched-tokens 1024 --gpu-memory-utilization 0.96 \
    --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,6,8,12,16]}' \
    --reasoning-parser qwen3 --enable-prompt-tokens-details --distributed-executor-backend uni \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --tool-strict-level parameter \
    --kv-transfer-config '{"kv_connector":"HostTierConnector","kv_connector_module_path":"sparsekv.tier_connector","kv_role":"kv_both"}' \
    --no-scheduler-reserve-full-isl --block-size 864 \
    --limit-mm-per-prompt '{"image":{"count":1}}' \
    --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152}}' \
    --long-prefill-token-threshold 256 --mamba-ssm-cache-dtype float16 \
    --speculative-config '{"method":"dflash","model":"/opt/orca/draft","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,1,5],[2,4,3]],"kv_cache_dtype":"int8_per_token_head","draft_load_config":{"load_format":"safetensors"},"attention_backend":"CUSTOM"}'
