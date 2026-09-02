#!/usr/bin/env bash
# GPU maintenance-window runbook: engine trial on the single RTX 3090.
# Phases: baseline (production vLLM via port-forward, no downtime) ->
#         suspend Flux + scale llama to 0 -> llama.cpp arms -> SGLang smoke ->
#         restore production. Every arm is time-boxed; failures never block restore.
set -uo pipefail

S=/tmp/claude-1000/-data-docker-services/69693a03-0b1b-4346-93f7-2117edd3c7d8/scratchpad/engines
BENCH=$S/bench
RESULTS=$BENCH/results.jsonl   # run-arm.sh appends here (fixed path)
LOG=$S/window.log
MODELS=/data/buttercup_6tb/k3s/llama-models
LLAMA_IMAGE=${LLAMA_IMAGE:-llama:trial-2026-09-02}
LLAMA_PORT=18081
ARM_TIMEOUT=${ARM_TIMEOUT:-3600}   # per arm, seconds
CTX=${CTX:-131072}
# Production-equivalent llama.cpp base args (from k8s configmap qwen3.8-27b entry, ctx lowered to fit the bigger drafters)
BASE_ARGS=(-m /models/Qwen3.8-27B-UD-Q4_K_XL-v3.gguf --mmproj /models/mmproj-Qwen3.8-27B-Q8_0.gguf
  --host 0.0.0.0 --port $LLAMA_PORT --ctx-size $CTX --predict $CTX
  --threads 8 --threads-batch 8 --parallel 1 --batch-size 512 --ubatch-size 512
  --no-mmap -fa on --jinja --cache-type-k q4_0 --cache-type-v q4_0
  --cache-type-k-draft q4_0 --cache-type-v-draft q4_0
  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 --repeat-penalty 1.0 -ngl 99 -ngld 99
  --alias qwen3.8-27b)
ENV_ARGS=(-e GGML_CUDA_MMVQ_NE11_MAX=3 -e GGML_CUDA_MMQ_SMALLN=3 -e 'LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"reasoning_effort":"medium"}')

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_health() { # url seconds
  local url=$1 t=${2:-900} i=0
  while ! curl -fsS -m 5 "$url" >/dev/null 2>&1; do sleep 5; i=$((i+5)); [ $i -ge $t ] && return 1; done; return 0
}

gpu_free() { # wait until <500 MiB used
  for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && return 0; sleep 5; done; return 1
}

SKIP_LLAMA=0
run_llama_arm() { # tag  extra-args...
  local tag=$1; shift
  [ "$SKIP_LLAMA" = 1 ] && { log "ARM $tag: skipped (validation failed)"; return 1; }
  log "ARM $tag: starting llama-server $*"
  docker rm -f trial-llama >/dev/null 2>&1
  docker run -d --name trial-llama --gpus all -p 127.0.0.1:$LLAMA_PORT:$LLAMA_PORT \
    -v $MODELS:/models:ro "${ENV_ARGS[@]}" --entrypoint /usr/local/bin/llama-server \
    "$LLAMA_IMAGE" "${BASE_ARGS[@]}" "$@" > /dev/null || { log "ARM $tag: docker run failed"; return 1; }
  if ! wait_health http://127.0.0.1:$LLAMA_PORT/health 900; then
    log "ARM $tag: never became healthy"; docker logs --tail 40 trial-llama | tee -a "$LOG"; docker rm -f trial-llama; return 1
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$LOG"
  timeout $ARM_TIMEOUT "$BENCH/run-arm.sh" "$tag" "http://127.0.0.1:$LLAMA_PORT/v1" 2>&1 | tee -a "$LOG"
  docker logs trial-llama 2>&1 | grep -iE 'accept|draft|n_draft|spec' | tail -5 | tee -a "$LOG"
  docker rm -f trial-llama >/dev/null; gpu_free
}

restore_production() {
  log "RESTORE: resuming Flux and scaling llama back"
  docker rm -f trial-llama sglang-qwen38-trial >/dev/null 2>&1
  kubectl -n apps scale deploy/llama --replicas=1
  kubectl -n apps scale deploy/llama-cache-canary --replicas=1 2>/dev/null
  flux resume kustomization apps
  kubectl -n apps rollout status deploy/llama --timeout=25m | tee -a "$LOG"
}
trap restore_production EXIT

phase=${1:-all}

if [ "$phase" = all ] || [ "$phase" = baseline ]; then
  log "PHASE baseline: production vLLM via port-forward (no downtime)"
  kubectl -n apps port-forward --address 127.0.0.1 svc/llama 28471:8080 >/dev/null 2>&1 & PF=$!; sleep 4
  timeout $ARM_TIMEOUT "$BENCH/run-arm.sh" vllm-prod-v10 http://127.0.0.1:28471/v1 2>&1 | tee -a "$LOG"
  kill $PF
  [ "$phase" = baseline ] && { trap - EXIT; exit 0; }
fi

log "PHASE takedown: suspend Flux, scale llama + canary to 0"
flux suspend kustomization apps
kubectl -n apps scale deploy/llama-cache-canary --replicas=0
kubectl -n apps scale deploy/llama --replicas=0
kubectl -n apps wait --for=delete pod -l app.kubernetes.io/name=llama --timeout=300s
gpu_free || log "WARN: GPU still shows memory in use"

# Kernel validation of the rebased patches (0005's swizzle-aware dequant has never executed):
# FLASH_ATTN_EXT + MUL_MAT conformance in the trial image, with the production env gates set.
log "PHASE validate: test-backend-ops in $LLAMA_IMAGE"
docker run --rm --gpus all "${ENV_ARGS[@]}" -e GGML_CUDA_FATTN_MMA_Q=1 --entrypoint /usr/local/bin/test-backend-ops \
  "$LLAMA_IMAGE" -b CUDA0 -o FLASH_ATTN_EXT 2>&1 | tail -3 | tee -a "$LOG"
docker run --rm --gpus all "${ENV_ARGS[@]}" --entrypoint /usr/local/bin/test-backend-ops \
  "$LLAMA_IMAGE" -b CUDA0 -o MUL_MAT 2>&1 | tail -3 | tee -a "$LOG"
if grep -qE 'FAIL|Backend CUDA0: FAIL' "$LOG"; then log "VALIDATION FAILED: skipping llama.cpp arms"; SKIP_LLAMA=1; fi

# llama.cpp arms (interleave a repeat of the production-equivalent arm at the end as a same-block control)
run_llama_arm llama-mtp-ngram   --model-draft /models/mtp-Qwen3.8-27B-Q4_0-d48k.gguf --spec-type draft-mtp,ngram-mod --spec-draft-n-max 5 --spec-ngram-mod-n-match 32
run_llama_arm llama-dflash2q8-n4 --model-draft /models/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 4
run_llama_arm llama-dflash2q8-n7-ngram --model-draft /models/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash,ngram-mod --spec-draft-n-max 7 --spec-ngram-mod-n-match 32
run_llama_arm llama-dspark-q8   --model-draft /models/Qwen3.8-27B-DSpark-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 5
run_llama_arm llama-mtp-only    --model-draft /models/mtp-Qwen3.8-27B-Q4_0-d48k.gguf --spec-type draft-mtp --spec-draft-n-max 5
run_llama_arm llama-mtp-ngram-ctrl --model-draft /models/mtp-Qwen3.8-27B-Q4_0-d48k.gguf --spec-type draft-mtp,ngram-mod --spec-draft-n-max 5 --spec-ngram-mod-n-match 32

if [ -x "$S/sglang/run-sglang.sh" ]; then
  # Order per prep notes: nospec isolates checkpoint/Marlin/template; stock tests whether
  # sglang#35822 (native MTP verify hang on sm86 with Qwen3.8-27B) reproduces; hybrid only if stock survives.
  log "PHASE sglang (time-boxed: nospec 40 min, stock 30 min, hybrid 40 min)"
  SG=http://127.0.0.1:18030/v1
  sg_arm() { # mode tag budget_s full|short   (run-sglang.sh is a foreground `docker run --rm`)
    local mode=$1 tag=$2 budget=$3 kind=$4 rc=1
    log "SGLANG $tag ($mode): launching"
    "$S/sglang/run-sglang.sh" "$mode" > "$S/sglang-$tag.log" 2>&1 & local lp=$!
    if timeout $((budget/2)) "$S/sglang/smoke.sh" 2>&1 | tee -a "$LOG"; then
      if [ "$kind" = full ]; then
        timeout $budget "$BENCH/run-arm.sh" "$tag" "$SG" 2>&1 | tee -a "$LOG" && rc=0
      else
        timeout $budget bash -c "cd $BENCH && python3 bench.py decode --base-url $SG --model qwen3.8-27b --tag $tag --reasoning medium --n 3 && python3 bench.py session --base-url $SG --model qwen3.8-27b --tag $tag --reasoning medium --turns 6" 2>&1 | tee -a "$LOG" && rc=0
      fi
    else
      log "SGLANG $tag: smoke failed"; tail -30 "$S/sglang-$tag.log" | tee -a "$LOG"
    fi
    docker rm -f sglang-qwen38-trial >/dev/null 2>&1; wait $lp 2>/dev/null; gpu_free; return $rc
  }
  sg_arm nospec sglang-nospec 2400 short && sg_arm stock sglang-mtp 1800 short && sg_arm hybrid sglang-hybrid 2400 full
fi

log "DONE arms; restoring production (trap)"
