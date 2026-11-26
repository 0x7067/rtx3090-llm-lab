#!/usr/bin/env bash
# Stage-1 flag sweep for NariQ on the RTX 3090 with llama-bench (no server, no sampling).
#
#   ./sweep.sh stage1     # 3.0bpw at depth 4096: threads x KV type x ubatch
#   ./sweep.sh stage2 ARGS...  # 3.0 and 2.5 bpw at depths 0/16K/64K with the chosen flags, e.g.
#                              #   ./sweep.sh stage2 -t 1 -ctk bf16 -ctv bf16 -ub 512
#
# The GPU must be free (this script never touches k8s). Output: sweep/<stage>-<bpw>.jsonl
set -euo pipefail
task_dir=$(cd "$(dirname "$0")" && pwd)
image=ghcr.io/nari-labs/nari-llama.cpp@sha256:1d7b7bb878a2d9381a977d7ea4a7d2aa7f0595f02905e8c5808c19a1bc111d79
weights=/data/models/qwen3.8/nari-labs/Qwen3.8-27B-NariQ-GGUF
out=$task_dir/sweep
mkdir -p "$out"
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || { echo "GPU busy" >&2; exit 1; }
nvidia-smi --query-gpu=driver_version,power.limit,clocks.max.sm,temperature.gpu --format=csv > "$out/gpu-state-$(date -u +%H%M%S).csv"

bench() {  # tag bpw args...
  local tag=$1 bpw=$2; shift 2
  docker run --rm --gpus all --cpus 8 --memory 32g -v "$weights:/models:ro" \
    --entrypoint /app/llama-bench "$image" -m "/models/Qwen3.8-27B-NariQ-${bpw}bpw.gguf" \
    -ngl 999 -fa 1 -r 3 -o jsonl "$@" | tee -a "$out/$tag-$bpw.jsonl"
}

case ${1:-} in
  stage1)
    bench stage1 3.0 -p 2048 -n 128 -d 4096 -t 1,4,8 -ctk bf16,q8_0 -ctv bf16,q8_0 -ub 512,2048 ;;
  stage2)
    shift
    for bpw in 3.0 2.5; do bench stage2 "$bpw" -p 2048 -n 512 -d 0,16384,65536 "$@"; done ;;
  *) sed -n 2,9p "$0"; exit 2 ;;
esac
