#!/usr/bin/env bash
# Launch SGLang serving Qwen3.8-27B-INT4-RedHatAI on one RTX 3090 (sm86, 24 GiB).
#
#   ./run-sglang.sh stock     # image as-is, native in-checkpoint MTP via EAGLE
#   ./run-sglang.sh hybrid    # + PR #36783 overlay (MTP + N-gram retrieval)
#   ./run-sglang.sh nospec    # no speculative decoding: the control arm
#
# The GPU is in production use. This script REFUSES to start unless the card is
# essentially free. Set FORCE=1 only when you have taken the other job down.
#
# Every knob below is an env var, so no editing is needed:
#   PORT=18030 CTX=65536 MFS=0.92 SSM_DTYPE=float32 KV_DTYPE=fp8_e5m2
#   EAGER=1        decode+prefill CUDA graphs off (the #36048 stable path)
#   REPLAYSSM=1    --enable-linear-replayssm-spec (frees the verify state pool)
#   DECODE_GRAPH=breakable|full|tc_piecewise|disabled
#   DRY_RUN=1      print the docker command and exit, touching nothing
#
# Sizing is predicted off-GPU by ./vram-budget.py; read NOTES.md before changing
# MFS, CTX, SSM_DTYPE or the radix-cache flags -- they interact non-obviously and
# the failure mode is a hard RuntimeError at boot, not a slowdown.
set -euo pipefail

MODE="${1:-stock}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- tunables ---
IMAGE="${IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260828-daf63171}"
HYBRID_IMAGE="${HYBRID_IMAGE:-sglang-hybrid-mtp-ngram:pr36783}"
MODEL_DIR="${MODEL_DIR:-/data/buttercup_6tb/k3s/vllm-trial/models/Qwen3.8-27B-INT4-RedHatAI}"
CHAT_TEMPLATE_HOST="${CHAT_TEMPLATE_HOST:-/data/docker-services/k8s/workloads/apps/llama/chat_template.jinja}"
PORT="${PORT:-18030}"
BIND_ADDR="${BIND_ADDR:-127.0.0.1}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b}"
CTX="${CTX:-65536}"
MFS="${MFS:-0.92}"
SSM_DTYPE="${SSM_DTYPE:-float32}"
KV_DTYPE="${KV_DTYPE:-fp8_e5m2}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-1024}"
MAX_RUNNING="${MAX_RUNNING:-1}"
JIT_CACHE="${JIT_CACHE:-$HERE/cache}"
CONTAINER_NAME="${CONTAINER_NAME:-sglang-qwen38-trial}"

# ------------------------------------------------------------- GPU guard -----
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
NEED_MIB="${NEED_MIB:-23000}"
if [[ "${FORCE:-0}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
  if (( FREE_MIB < NEED_MIB )); then
    echo "REFUSING TO START: only ${FREE_MIB} MiB free on the GPU, need >= ${NEED_MIB} MiB." >&2
    echo "The weights alone are 18.12 GiB. Something else is using the card:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    echo "Stop it first, or re-run with FORCE=1 if you know better." >&2
    exit 1
  fi
fi

# ------------------------------------------------------------ preflight ------
[[ -d "$MODEL_DIR" ]] || { echo "no such model dir: $MODEL_DIR" >&2; exit 1; }
[[ -f "$MODEL_DIR/model_mtp.safetensors" ]] || {
  echo "WARNING: $MODEL_DIR has no model_mtp.safetensors; MTP modes will fail." >&2; }
[[ -f "$CHAT_TEMPLATE_HOST" ]] || { echo "no chat template: $CHAT_TEMPLATE_HOST" >&2; exit 1; }
mkdir -p "$JIT_CACHE"

# Docker 29 on this host exposes the GPU through CDI. Fall back to the legacy
# runtime flag if no CDI device is registered.
if docker info -f '{{json .}}' 2>/dev/null | grep -q 'nvidia.com/gpu' \
   || [[ -e /etc/cdi/nvidia.yaml || -e /var/run/cdi/nvidia.yaml ]]; then
  GPU_FLAG=(--device nvidia.com/gpu=all)
else
  GPU_FLAG=(--gpus all)
fi

# -------------------------------------------------------------- arg build ----
ARGS=(
  --model-path /models/qwen38
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0 --port 30000
  --context-length "$CTX"
  --mem-fraction-static "$MFS"
  --dtype bfloat16
  --quantization compressed-tensors
  --kv-cache-dtype "$KV_DTYPE"
  --mamba-ssm-dtype "$SSM_DTYPE"
  --linear-attn-backend triton
  --chunked-prefill-size "$CHUNKED_PREFILL"
  --max-running-requests "$MAX_RUNNING"
  --disable-radix-cache
  --chat-template /chat_template.jinja
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --trust-remote-code
)

# Why --disable-radix-cache is not optional here: it is the only setting that
# takes the GDN state-slot multiplier (_calculate_mamba_ratio) to 1. At the
# default 5 (or 3 with no_buffer) the solver lands on max_mamba_cache_size=0
# and the server dies with "Hybrid (mamba/linear-attention) state cache is too
# small to serve any requests" -- issue #36048's exact failure. See NOTES.md.

# Prefill CUDA graph capture hangs on this hybrid-GDN architecture (#36048 #35437).
ARGS+=(--disable-prefill-cuda-graph)

if [[ "${EAGER:-0}" == "1" ]]; then
  ARGS+=(--cuda-graph-backend-decode disabled)
elif [[ -n "${DECODE_GRAPH:-}" ]]; then
  ARGS+=(--cuda-graph-backend-decode "$DECODE_GRAPH")
fi

# Cookbook MTP operating point for Qwen3.8-27B: 3 steps, chain (topk 1), 4 tokens.
MTP_ARGS=(
  --speculative-algorithm EAGLE
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
)

RUN_IMAGE="$IMAGE"
DEFAULT_REPLAYSSM=0
case "$MODE" in
  nospec)
    ;;
  stock)
    ARGS+=("${MTP_ARGS[@]}")
    ;;
  hybrid)
    RUN_IMAGE="$HYBRID_IMAGE"
    # At L=9 the per-request verify-intermediate reservation is
    # 9 x 153.9 MB x (reqs+1), which leaves only ~24k KV tokens on a 24 GiB
    # card (vram-budget.py row G). ReplaySSM keeps the verify intermediates on
    # a fixed ring instead of per-request slots, taking that term to zero and
    # the KV pool back to ~109k tokens (row I). Set REPLAYSSM=0 to opt out,
    # but then also drop CTX to 16384 or set SSM_DTYPE=bfloat16 (row H).
    DEFAULT_REPLAYSSM=1
    docker image inspect "$RUN_IMAGE" >/dev/null 2>&1 || {
      echo "hybrid image $RUN_IMAGE not built. Run:" >&2
      echo "  $HERE/build-hybrid.sh" >&2
      exit 1; }
    # L (num_draft_tokens) must exceed num_steps+1 so retrieval has slots to
    # fill; 9 is the value in the PR's own docs.
    STEPS="${HYBRID_STEPS:-3}"
    TAU="${HYBRID_TAU:-off,off,0.40,0.55}"
    # The tau vector is indexed by verify-chain column and needs exactly
    # num_steps+1 entries; column 0 (the bonus token) and column 1 (whose
    # logprob comes from the previous draft-extend) must both be disabled.
    # parse_position_thresholds() raises on any violation, but only after the
    # 18 GiB weight load, so check the arity here where it costs nothing.
    NTAU=$(awk -F, '{print NF}' <<<"$TAU")
    if (( NTAU != STEPS + 1 )); then
      echo "HYBRID_TAU has $NTAU entries but HYBRID_STEPS=$STEPS needs $((STEPS+1))." >&2
      echo "  got: $TAU" >&2
      exit 1
    fi
    case "$(cut -d, -f1 <<<"$TAU")" in off|disabled|none|0|"") ;;
      *) echo "HYBRID_TAU column 0 must be off (it is the bonus token): $TAU" >&2; exit 1 ;; esac
    case "$(cut -d, -f2 <<<"$TAU")" in off|disabled|none|0|"") ;;
      *) echo "HYBRID_TAU column 1 must be off (its logprob is one iteration stale): $TAU" >&2; exit 1 ;; esac
    ARGS+=(
      --speculative-algorithm EAGLE
      --speculative-num-steps "$STEPS"
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens "${HYBRID_L:-9}"
      --speculative-hybrid-retrieval
      --speculative-hybrid-tau-per-pos "$TAU"
    )
    [[ "${HYBRID_INDEX_PROMPT:-1}" == "1" ]] && ARGS+=(--speculative-hybrid-index-prompt)
    [[ "${HYBRID_FULL_GRAPH:-0}" == "1" ]] && ARGS+=(--speculative-hybrid-full-cuda-graph)
    [[ "${HYBRID_DYNAMIC_TAU:-0}" == "1" ]] && ARGS+=(--speculative-hybrid-dynamic-tau)
    # NOT set: --speculative-hybrid-ragged (DeepSeek-V4 attention backend only,
    # asserts at startup on any other backend) and --speculative-hybrid-overlap
    # (leaving it off lets the hook force --disable-overlap-schedule, which is
    # the path the PR actually validated).
    ;;
  *)
    echo "usage: $0 {stock|hybrid|nospec}" >&2; exit 1 ;;
esac

if [[ "${REPLAYSSM:-$DEFAULT_REPLAYSSM}" == "1" && "$MODE" != "nospec" ]]; then
  ARGS+=(--enable-linear-replayssm-spec)
fi

# ------------------------------------------------------------------ launch ---
DOCKER_ARGS=(
  run --rm --name "$CONTAINER_NAME"
  "${GPU_FLAG[@]}"
  --ipc=host --shm-size=16g
  -p "${BIND_ADDR}:${PORT}:30000"
  -v "$MODEL_DIR:/models/qwen38:ro"
  -v "$CHAT_TEMPLATE_HOST:/chat_template.jinja:ro"
  -v "$JIT_CACHE:/root/.cache"
  # Both are required for a stable request path on this checkpoint: the GDN
  # conv-state cache is written by kernels whose dtype must match the cache, or
  # the first request dies in gdn_backend.py with "Index put requires the source
  # and destination dtypes match" (#36048).
  -e SGLANG_MAMBA_CONV_DTYPE="${MAMBA_CONV_DTYPE:-bfloat16}"
  -e SGLANG_MAMBA_SSM_DTYPE="$SSM_DTYPE"
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
)
[[ -n "${EXTRA_ENV:-}" ]] && DOCKER_ARGS+=(-e "$EXTRA_ENV")

CMD=(docker "${DOCKER_ARGS[@]}" "$RUN_IMAGE"
     python3 -m sglang.launch_server "${ARGS[@]}")

echo "mode=$MODE image=$RUN_IMAGE"
echo "api  http://${BIND_ADDR}:${PORT}/v1   model=$SERVED_NAME"
echo
printf '%q ' "${CMD[@]}"; echo; echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN=1, nothing launched]"; exit 0
fi
exec "${CMD[@]}"
