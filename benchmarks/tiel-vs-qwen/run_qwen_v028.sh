#!/usr/bin/env bash
# Serve Qwen3.8-27B-W4A16-AutoRound-fast on stock vLLM 0.28.0 (port 8095).
# Purpose: A/B the fused GDN MTP decode kernel (PR #51674) against our 0.27.1 baseline.
#
# READ FIRST — two verified blockers (see report):
#  1. The checkpoint carries mtp.draft_lm_head.* (our local draft-vocab-head patch).
#     Stock 0.28.0 remaps mtp.* -> model.* and has no draft_lm_head module, so weight
#     loading raises ValueError. This script WILL fail to load until that is resolved.
#  2. Even if it loaded, the fused kernel's runtime gate requires num_v_heads ==
#     8 * num_k_heads (qwen_gdn_linear_attn.py:1808); this model is 48 v / 16 k
#     (ratio 3), so the kernel never engages. Confirmed: num_v_heads/num_k_heads are
#     never reassigned between :374-375 and :1808 (every other use divides by tp_size
#     explicitly), so this is the raw config ratio.
set -euo pipefail

ROOT=${ROOT:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"}
MODEL=/data/buttercup_6tb/k3s/vllm-trial/models/Qwen3.8-27B-W4A16-AutoRound-fast

# Scratch caches, mirroring the container's isolation without changing HOME.
V028_CACHE_ROOT=${V028_CACHE_ROOT:-"$ROOT/v028-cache"}
export HF_HOME="$V028_CACHE_ROOT/hf"
export TORCH_HOME="$V028_CACHE_ROOT/torch"
export XDG_CACHE_HOME="$V028_CACHE_ROOT"
export VLLM_CACHE_ROOT="$V028_CACHE_ROOT/vllm"
export HF_HUB_OFFLINE=1
mkdir -p "$HF_HOME" "$TORCH_HOME" "$VLLM_CACHE_ROOT"

# The knob under test. "cuda" is already the 0.28.0 default, but setting it
# EXPLICITLY turns an unsupported config into a startup ValueError instead of a
# silent Triton fallback (envs.py:1213-1221, qwen_gdn_linear_attn.py:494-503) —
# that is what we want for a benchmark. Set to "triton" for the control arm.
export VLLM_GDN_DECODE_KERNEL="${VLLM_GDN_DECODE_KERNEL:-cuda}"

V028_VLLM_BIN=${V028_VLLM_BIN:-"$ROOT/v028-env/bin/vllm"}
exec "$V028_VLLM_BIN" serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 8095 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 140000 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype float16 \
  --mamba-cache-mode align \
  --no-async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"max_cudagraph_capture_size":32,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --chat-template /data/development/projects/docker-services/k8s/workloads/apps/llama/chat_template.jinja

# Flag notes (each verified against this venv's `vllm serve --help=all`):
#  --mamba-cache-mode align  ADDED. Qwen3_5MTP raises NotImplementedError on
#      mamba_cache_mode="all" (qwen3_5_mtp.py:225-229). Prefix caching now
#      Explicit because Qwen3_5MTP hard-raises on "all". Do not set "all".
#  --enable-prefix-caching   KEPT but now a no-op-ish: the default is None and
#      resolves to on for mamba/hybrid models in 0.28.0 (#50991). Harmless, and
#      it documents intent.
#  --no-async-scheduling     REQUIRED. "mtp" is in EagleModelTypes, so
#      vllm.py:1185-1234 resolves async_scheduling to True by default. Both
#      --async-scheduling and --no-async-scheduling exist in 0.28.0.
#  --mamba-ssm-cache-dtype float16   KEPT to match the baseline, but note it
#      DISABLES the fused kernel ON ITS OWN, and needlessly: the checkpoint already
#      declares mamba_ssm_dtype="float32", which IS an accepted state dtype. The
#      float16 override is our benchmark choice, not a property of the model.
#      FUSED_GDN_STATE_DTYPES is
#      (float32, bfloat16) (qwen_gdn_linear_attn.py:90), checked at
#      qwen_gdn_linear_attn.py:523 and again at :1807. For a fused-kernel run you
#      must drop this flag (or use bfloat16) — otherwise VLLM_GDN_DECODE_KERNEL=cuda
#      raises at startup. Kept here because baseline parity was the stated goal;
#      change it deliberately, not by accident.
#  draft_sample_method       NOT dropped-as-unsupported: it DOES exist in stock
#      0.28.0's SpeculativeConfig. It is simply not set here, matching the flags
#      handed to me. Add it if the baseline sets it.
#  Local 0.27.1 patches with NO stock equivalent: the draft vocab head
#      (mtp.draft_lm_head + mtp_draft_vocab_ids.pt) and our sampler changes.
#      Stock has use_heterogeneous_vocab / use_local_argmax_reduction /
#      rejection_sample_method in SpeculativeConfig, which are adjacent but not
#      the same thing. Treat any result as engine-vs-engine, not apples-to-apples.
