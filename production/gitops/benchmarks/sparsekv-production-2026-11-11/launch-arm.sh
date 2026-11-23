#!/usr/bin/env bash
# Start one benchmark arm after production has released the GPU.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
arm=${1:-}
mode=${2:-run}
case "$arm" in
  baseline|candidate) ;;
  *) echo "Usage: $0 baseline|candidate [--check]" >&2; exit 2 ;;
esac
case "$mode" in
  run) runtime=(--net-host --runc-binary /usr/bin/nvidia-container-runtime) ;;
  --check) runtime=(--env PRINT_ARGV=1 --env CUDA_VISIBLE_DEVICES=) ;;
  *) echo "Usage: $0 baseline|candidate [--check]" >&2; exit 2 ;;
esac
scratch=$(cat "$here/.scratch")
production="$here/production"
image=127.0.0.1:5000/home-llm-strata@sha256:83077f60377d9e303d446a2f045381daab28a123e8e314c5bbe1ba53314884a2
models=/srv/llama-nvme/hyperqwen-bakeoff/models
site=/app/venv/lib/python3.12/site-packages
cache="$scratch/$arm-cache"
mkdir -p "$cache"

common=(
  --rm "${runtime[@]}"
  --cpus 12 --memory-limit 38654705664
  --mount 'type=tmpfs,src=tmpfs,dst=/dev/shm,options=size=24g:mode=1777'
  --mount "type=bind,src=$models,dst=/app/models,options=rbind:ro"
  --mount "type=bind,src=$cache,dst=/cache,options=rbind:rw"
  --mount "type=bind,src=$production/chat_template.jinja,dst=/chat-template/chat_template.jinja,options=rbind:ro"
  --env HOME=/cache --env XDG_CACHE_HOME=/cache --env HF_HOME=/cache/huggingface
  --env TORCH_EXTENSIONS_DIR=/cache/torch_extensions --env HF_HUB_OFFLINE=1
  --env NVIDIA_VISIBLE_DEVICES=all --env 'NVIDIA_DRIVER_CAPABILITIES=compute,utility'
  --env VLLM_NO_USAGE_STATS=1 --env DO_NOT_TRACK=1 --env PYTHONUNBUFFERED=1 --env PYTHONPATH=
  --env HOST=127.0.0.1 --env VLLM_API_KEY=sparsekv-bench-local
  --env VLLM_OFFLOAD_KEEP_SHM=1
  --env MODEL=/app/models/Qwen3.8-27B-W4A16-AutoRound
  --env INT8_ACT=int8 --env INT8_LAYERS=mlp --env EMBED_UVA=1
  --env CTX=long --env PTH_FA=1 --env MAX_SEQS=4 --env PREFIX_CACHE=1
  --env VISION=1 --env VISION_OFFLOAD=1
)

if [[ $arm == baseline ]]; then
  options=(
    --mount "type=bind,src=$production/start_qwen_cache.py,dst=/chat-template/start_qwen_cache.py,options=rbind:ro"
    --env PORT=5890 --env SPEC=dflash2 --env DFLASH_TOKENS=7 --env LOOKUP=1
    --env DFLASH_MAX_LEN=200000 --env KV_MEM=7784628224
    --env FA2_FP8KV=0
    --env PTH_CPU_CACHE=1 --env KV_OFFLOAD_GB=16 --env PREFIX_RETENTION=5184
    --env CUDAGRAPH_MODE=PIECEWISE
    --env 'EXTRA_ARGS=--chat-template /chat-template/chat_template.jinja --long-prefill-token-threshold 864 --long-prefill-token-threshold-adaptive --watermark 0.1'
  )
  command=(/usr/bin/setpriv --reuid=1000 --regid=1000 --clear-groups --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs /app/venv/bin/python -I -S /chat-template/start_qwen_cache.py)
else
  [[ -s $scratch/overlay.tar ]] || { echo 'Candidate overlay is missing' >&2; exit 1; }
  options=(
    --mount "type=bind,src=$scratch/hq,dst=/sparsebench,options=rbind:ro"
    --mount "type=bind,src=$scratch/overlay.tar,dst=/overlay.tar,options=rbind:ro"
    --env PORT=18091 --env BIND=127.0.0.1 --env HQ_GPU=0 --env HQ_TMP=/cache/tmp
    --env HQ_NAMES=qwen3.8-27b --env SPEC=mtp --env DRAFT_TOKENS=7
    --env "MAX_LEN=${SPARSE_BENCH_MAX_LEN:-200000}"
    --env GPU_UTIL=0.94 --env SPARSE_PROFILE=tier --env SPARSE_TIER_HOT_FRAC=0.025
    --env SPARSE_TIER_HOST_GB=17.8 --env SPARSE_TIER_HOST_BITS=4 --env SPARSE_SKETCH_BITS=4
    --env SPARSE_STATS_SUB=1 --env SPARSE_SELECT_EVERY=1
    --env 'EXTRA_ARGS=--chat-template /chat-template/chat_template.jinja'
  )
  # Expansion occurs inside the disposable container, not in this shell.
  # shellcheck disable=SC2016
  command=(/bin/bash -euc '
    tar -xf /overlay.tar -C "$1" --exclude=share/sparsekv --exclude="*/__pycache__/*"
    tar -xf /overlay.tar -C /app/venv share/sparsekv
    exec setpriv --reuid=1000 --regid=1000 --clear-groups --bounding-set=-all \
      --inh-caps=-all --ambient-caps=-all --no-new-privs bash /sparsebench/serve-sparse.sh
  ' -- "$site")
fi

exec sudo k3s ctr -n k8s.io run "${common[@]}" "${options[@]}" \
  "$image" "sparsekv-bench-$arm" "${command[@]}"
