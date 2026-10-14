#!/usr/bin/env bash
# Qwen3.8 engine bake-off: one arm at a time, isolated loopback containers.
#
#   ./run-bakeoff.sh                  # all arms, in ARMS order
#   ./run-bakeoff.sh sglang_exl3 ...  # a subset
#
# Before running: the GPU must be free (production llama scaled to 0 by you;
# this script never touches k8s). GPU_PROFILE=cool (default) applies the
# matched-run profile from ../local-llm-reset-2026-09-30/gpu-cool-profile.sh via
# sudo (it will prompt) and restores the normal profile on exit;
# GPU_PROFILE=normal runs under the host's normal profile and records that.
# An arm that fails to start or run is recorded in results/<arm>/STATUS and the
# script moves on to the next arm.
set -uo pipefail

task_dir=$(cd "$(dirname "$0")" && pwd)
harness=$task_dir/../paired-harness/harness.py
plan=$task_dir/plan.json
profile_script=$task_dir/../local-llm-reset-2026-09-30/gpu-cool-profile.sh
gpu_profile=${GPU_PROFILE:-cool}
results=$task_dir/results

llama_image=ghcr.io/ggml-org/llama.cpp@sha256:c4348afb91499b10fad12f92626c7a8ca8b69d6475d10c793b810cb61237ce00
hyperqwen_image=ghcr.io/syv-ai/hyperqwen@sha256:e99df7e6dc73208971ae0d1f1063deb100c997308248f253174176cf0514cb9e
sglang_image=ghcr.io/0xsero/sglang-exl3@sha256:c8922bd7256caf1b273af7ac49221e652be9e6072c1cdec4b959f506fd415d79
strata_image=local/strata-bakeoff:v0.1.44.1

q4ks_dir=/data/models/qwen3.8/bartowski/Qwen3.8-27B-GGUF
gsq_dir=/data/models/qwen3.8/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF
exl3_dir=/data/models/qwen3.8/turboderp/Qwen3.8-27B-exl3/6fe61ad620abfe97c5b49f9722c2bceeea4ccc28-3.00bpw
hyperqwen_root=/data/models/hyperqwen-bakeoff
# Strata reads the n-gram shard and (LOW_RAM=on) the expert pack from disk while it
# answers, so both live on the SSD root volume; /data/models is a spinning HDD.
flashnext_dir=/data/bakeoff-ssd/flash-next-IQ3_XXS
strata_data=/data/bakeoff-ssd/strata-data

ARMS=(llamacpp_q4ks llamacpp_gsq vllm_hyperqwen sglang_exl3 strata_flashnext_iq3xxs)
(($#)) && ARMS=("$@")

active_container=
profile_applied=false

cleanup() {
  [[ -z $active_container ]] || docker rm -f "$active_container" >/dev/null 2>&1 || true
  if [[ $profile_applied == true ]]; then
    sudo bash "$profile_script" restore || echo "WARNING: GPU profile restore failed; run: sudo bash $profile_script restore" >&2
  fi
}
trap cleanup EXIT INT TERM

die() { echo "precheck: $*" >&2; exit 1; }

# ---- prechecks (no side effects) -------------------------------------------
[[ -f $plan ]] || die "missing $plan (run ./build-plan.py)"
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || die "GPU has compute processes; free it first"
systemctl is-active --quiet vram-thermal-guard.service || die "vram-thermal-guard inactive"
systemctl is-active --quiet vram-fan-curve.service || die "vram-fan-curve inactive"
[[ ! -e /run/gpu-thermal-trip ]] || die "/run/gpu-thermal-trip present"
for p in 18180 18181 18182 18183; do
  ! curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$p/" || die "port $p already in use"
done
mkdir -p "$results"

if [[ $gpu_profile == cool ]]; then
  sudo bash "$profile_script" apply || die "could not apply GPU profile"
  profile_applied=true
fi
nvidia-smi --query-gpu=driver_version,power.limit,clocks.max.sm --format=csv > "$results/gpu-profile-$gpu_profile.txt"

# ---- helpers ---------------------------------------------------------------
wait_ready() {  # container url attempts(5 s each)
  local i
  for i in $(seq 1 "$3"); do
    curl -fsS --max-time 5 "$2" >/dev/null 2>&1 && return 0
    [[ $(docker inspect "$1" --format '{{.State.Running}}' 2>/dev/null) == true ]] || return 1
    sleep 5
  done
  return 1
}

gpu_telemetry() {  # container out
  while [[ $(docker inspect "$1" --format '{{.State.Running}}' 2>/dev/null) == true ]]; do
    [[ -e /run/gpu-thermal-trip ]] && trip=true || trip=false
    printf '%s\ttrip=%s\t%s\n' "$(date -u +%FT%TZ)" "$trip" \
      "$(nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu,clocks.sm --format=csv,noheader,nounits)" >> "$2"
    sleep 5
  done
}

start_arm() {  # arm -> sets active_container, url, health, ready_attempts, max_ctx, model
  local arm=$1
  active_container=bake-$arm
  docker rm -f "$active_container" >/dev/null 2>&1 || true
  case $arm in
    llamacpp_q4ks)
      port=18181 health=/health ready_attempts=120 max_ctx=135168 model=q4ks-qwen3.8-27b
      docker run -d --name "$active_container" --gpus all -p 127.0.0.1:$port:$port \
        --memory 32g --memory-swap 32g --cpus 8 -v "$q4ks_dir:/models:ro" "$llama_image" \
        --model /models/Qwen3.8-27B-Q4_K_S.gguf --mmproj /models/mmproj-Qwen3.8-27B-f16.gguf \
        --ctx-size $max_ctx --parallel 1 --n-gpu-layers 999 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 --fit off --jinja \
        --spec-type draft-mtp --spec-draft-n-max 2 \
        --host 0.0.0.0 --port $port --alias $model --threads 10 ;;
    llamacpp_gsq)
      port=18181 health=/health ready_attempts=120 max_ctx=200704 model=gsq-qwen3.8-27b
      docker run -d --name "$active_container" --gpus all -p 127.0.0.1:$port:$port \
        --memory 32g --memory-swap 32g --cpus 8 -v "$gsq_dir:/models:ro" "$llama_image" \
        --model /models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --mmproj /models/mmproj-Qwen3.8-27B-BF16.gguf \
        --ctx-size $max_ctx --parallel 1 --n-gpu-layers 999 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 --fit off --jinja \
        --spec-type draft-mtp --spec-draft-n-max 2 \
        --host 0.0.0.0 --port $port --alias $model --threads 10 ;;
    vllm_hyperqwen)
      # First boot runs docker/prepare.sh (requantize into $hyperqwen_root/models) and
      # torch.compile/FlashInfer JIT, so allow up to 60 min.
      port=18180 health=/health ready_attempts=720 max_ctx=150000 model=qwen3.8-27b
      docker run -d --name "$active_container" --gpus all --ipc host -p 127.0.0.1:$port:$port \
        --memory 44g --memory-swap 44g --cpus 8 \
        -v "$hyperqwen_root/models:/app/models" -v "$hyperqwen_root/cache:/cache" \
        -e HOME=/cache -e SPEC=mtp -e CTX=long -e PREFIX_CACHE=1 -e PORT=$port \
        "$hyperqwen_image" single ;;
    sglang_exl3)
      port=18183 health=/health ready_attempts=360 max_ctx=212992 model=turboderp-Qwen3.8-27B-exl3-3.00bpw
      docker run -d --name "$active_container" --gpus all --shm-size 16g -p 127.0.0.1:$port:$port \
        --memory 44g --memory-swap 44g --cpus 8 \
        -v "$exl3_dir:/models/turboderp-Qwen3.8-27B-exl3-3.00bpw:ro" \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SGLANG_EXL3_EMBED_HOST=1 \
        -e SGLANG_EXL3_PREFILL_FP16_ACC=1 -e SGLANG_EXL3_VIT_MLP_CHUNK=8192 -e SGLANG_EXL3_VIT_SDPA=1 \
        -e SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 -e SGLANG_VLM_CACHE_SIZE_MB=256 \
        --entrypoint /opt/entrypoint.sh "$sglang_image" \
        python3 -m sglang.launch_server --model-path /models/turboderp-Qwen3.8-27B-exl3-3.00bpw \
        --quantization exl3 --trust-remote-code --host 0.0.0.0 --port $port \
        --served-model-name $model --context-length $max_ctx --mem-fraction-static 0.82 \
        --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --max-running-requests 1 \
        --max-mamba-cache-size 3 --cuda-graph-max-bs-decode 1 --cuda-graph-max-bs-prefill 240 \
        --chunked-prefill-size 4096 --max-prefill-tokens 4096 --prefill-decode-interval 16 \
        --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
        --mamba-radix-cache-strategy no_buffer --disable-overlap-schedule --image-processor-backend pil \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 4 \
        --speculative-token-map /opt/sglang-exl3/tokenmaps/qwen38_hot32k_v2.pt ;;
    strata_flashnext_iq3xxs)
      # GGUF_DIR must be writable on first start (setup writes <shard>.done after hashing).
      port=18182 health=/health ready_attempts=720 max_ctx=150000 model=qwen3.8-flash-next
      mkdir -p "$strata_data"
      docker run -d --name "$active_container" --gpus all -p 127.0.0.1:$port:$port \
        --memory 44g --memory-swap 44g --cpus 8 --ulimit memlock=-1:-1 --stop-timeout 60 \
        -v "$strata_data:/data" -v "$flashnext_dir:/gguf" \
        -e GGUF_DIR=/gguf -e FAMILY=qwen -e MODEL=IQ3_XXS -e CONTEXT=$max_ctx -e KV=int8 \
        -e LOW_RAM=on -e VISION=no -e GPU=0 -e HOST=0.0.0.0 -e PORT=$port \
        "$strata_image" ;;
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac
}

run_arm() {
  local arm=$1 out=$results/$1 status=ok
  if [[ -e $out ]]; then echo "skip $arm: $out exists (move it away to rerun)"; return 0; fi
  mkdir -p "$out"
  echo "== $arm $(date -u +%FT%TZ)"
  if ! start_arm "$arm" > "$out/container-id" 2> "$out/start.err"; then
    echo "start-failed" > "$out/STATUS"; active_container=; return 0
  fi
  docker inspect "$active_container" --format '{{.Image}}' > "$out/image-id"
  bash "$task_dir/monitor.sh" "$active_container" "$out/host-telemetry.tsv" > "$out/monitor.log" 2>&1 &
  local monitor_pid=$!
  gpu_telemetry "$active_container" "$out/gpu-telemetry.tsv" &
  local gpu_pid=$!

  if ! wait_ready "$active_container" "http://127.0.0.1:$port$health" "$ready_attempts"; then
    status=not-ready
  else
    curl -fsS "http://127.0.0.1:$port/v1/models" > "$out/models.json" 2>&1 || true
    python3 "$harness" run --plan "$plan" --arm "$arm" --out "$out/quality-run" --timeout 1800 \
      > "$out/harness.log" 2>&1 || status=harness-failed
    if [[ $(docker inspect "$active_container" --format '{{.State.Running}}' 2>/dev/null) == true ]]; then
      python3 "$task_dir/speed-probe.py" --arm "$arm" --base-url "http://127.0.0.1:$port/v1" \
        --model "$model" --max-ctx "$max_ctx" --out "$out/speed.json" > "$out/speed.log" 2>&1 \
        || status="${status/ok/}speed-errors"
    fi
  fi
  docker logs "$active_container" > "$out/server.log" 2>&1 || true
  docker stop -t 60 "$active_container" >/dev/null 2>&1 || true
  wait "$monitor_pid" 2>/dev/null; wait "$gpu_pid" 2>/dev/null
  docker rm "$active_container" >/dev/null 2>&1 || true
  active_container=
  grep -q '^abort' "$out/host-telemetry.tsv" 2>/dev/null && status="$status,memory-abort"
  grep -q 'trip=true' "$out/gpu-telemetry.tsv" 2>/dev/null && status="$status,thermal-trip"
  [[ -d $out/quality-run ]] && python3 "$harness" summarize --plan "$plan" --run "$out/quality-run" \
    --out "$out/summary.json" --markdown "$out/summary.md" > "$out/summarize.log" 2>&1 || true
  echo "$status" > "$out/STATUS"
  echo "   $arm: $status"
  # Stop the whole campaign on a thermal trip; let the guard's state be inspected.
  [[ $status != *thermal-trip* ]] || { echo "thermal trip; stopping" >&2; exit 2; }
  # Let the GPU and page cache settle between arms.
  sleep 30
}

for arm in "${ARMS[@]}"; do run_arm "$arm"; done
python3 "$task_dir/aggregate.py" > "$results/aggregate.log" 2>&1 || true
echo "done; see $results"
