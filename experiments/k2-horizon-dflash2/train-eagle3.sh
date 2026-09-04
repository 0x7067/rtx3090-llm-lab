#!/usr/bin/env bash
# Offline colocated EAGLE-3 training for K2-Horizon-7B on the RTX 3090.
# Reads the features captured by capture-eagle3.sh; the target is not loaded.
set -euo pipefail
W=/data/buttercup_6tb/specforge-work
E=/data/docker-services/rtx3090-llm-lab/experiments/k2-horizon-dflash2
export HF_HOME="$W/hf-home" UV_CACHE_DIR="$W/.uv-cache"
export CUDA_HOME="$W/venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK ${NVCC_PREPEND_FLAGS:-}"
cd "$W/SpecForge"; source "$W/venv/bin/activate"
specforge train --config "$E/train-k2-7b-eagle3-offline.yaml" \
  model.vocab_mapping_path="$W/cache/hidden_states/k2-7b-eagle3/vocab_mapping/vocab_mapping.pt" \
  "$@"
echo TRAIN-DONE
