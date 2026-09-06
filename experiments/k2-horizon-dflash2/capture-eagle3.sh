#!/usr/bin/env bash
# Offline EAGLE-3 feature capture for K2-Horizon-7B on the RTX 3090.
# Target bf16 (18 GB) alone on the GPU; llama-swap must be unloaded first.
# Only the last assistant turn is supervised: the mined agent contexts carry
# history turns written by other models, and training on those is what made
# the first pass worthless. 8,192 max-length is the smallest cap that leaves
# every sampled row with supervision (2,048 blanks 11%); the capture script
# sizes the SGLang KV pool at batch x max-length, so batch 2 is the ceiling
# that still fits beside the 18 GB bf16 target on a 24 GB card.
set -euo pipefail
W=/data/buttercup_6tb/specforge-work
E=/data/docker-services/rtx3090-llm-lab/experiments/k2-horizon-dflash2
export HF_HOME="$W/hf-home" UV_CACHE_DIR="$W/.uv-cache"
export CUDA_HOME="$W/venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
# The venv ships nvcc 13.3 but torch's CUDA runtime headers are 13.0; CCCL
# refuses the mix at compile time. The combination (newer compiler, older
# headers) is otherwise fine for the sm86 JIT kernels, so disable the check.
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK ${NVCC_PREPEND_FLAGS:-}"
cd "$W/SpecForge"; source "$W/venv/bin/activate"
NUM_SAMPLES="${NUM_SAMPLES:-}"
extra=(); [ -n "$NUM_SAMPLES" ] && extra+=(--num-samples "$NUM_SAMPLES")
torchrun --standalone --nproc_per_node 1 scripts/prepare_hidden_states.py \
  --strategy eagle3 \
  --target-model-path "$W/models/IFM/K2-Horizon-7B" \
  --draft-model-config "$E/configs/k2-horizon-7b-eagle3.json" \
  --trust-remote-code \
  --data-path "$W/cache/dataset/k2-eagle3-regenerated.jsonl" \
  --output-path "${OUTPUT_DIR:-$W/cache/hidden_states/k2-7b-eagle3-regen}" \
  --chat-template k2-horizon-thinking \
  --max-length 8192 \
  --train-only-last-turn \
  --tp-size 1 \
  --batch-size "${BATCH_SIZE:-2}" \
  --cache-dir "$W/cache" \
  --sglang-attention-backend "${ATTN_BACKEND:-triton}" \
  --sglang-mem-fraction-static "${MEM_FRACTION:-0.88}" \
  --sglang-context-length 8448 \
  "${extra[@]}"
echo CAPTURE-DONE
