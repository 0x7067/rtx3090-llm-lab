#!/usr/bin/env bash
# NariQ campaign (2026-11-25): run-bakeoff.sh from ../qwen38-bakeoff-2026-10-14 with
# NariQ arms and a same-day control arm. One arm at a time, isolated loopback containers.
#
#   ./run-nariq.sh                    # all arms, in ARMS order
#   ./run-nariq.sh nariq_3p0 ...      # a subset
#
# Before running: the GPU must be free (production llama scaled to 0 by you;
# this script never touches k8s). Runs under the host's normal GPU profile only:
# the 2026-10-14 "cool" profile needs vram-thermal-guard and gpu-mem-oc, which were
# removed from the host on 2026-11-02, so it can't be reproduced.
# NQ_FLAGS overrides the NariQ launch flags chosen from sweep.sh.
# An arm that fails to start or run is recorded in results/<arm>/STATUS and the
# script moves on to the next arm.
set -uo pipefail

task_dir=$(cd "$(dirname "$0")" && pwd)
harness=$task_dir/../paired-harness/harness.py
plan=$task_dir/plan.json
gpu_profile=normal
results=$task_dir/results

llama_image=ghcr.io/ggml-org/llama.cpp@sha256:c4348afb91499b10fad12f92626c7a8ca8b69d6475d10c793b810cb61237ce00
nariq_image=ghcr.io/nari-labs/nari-llama.cpp@sha256:1d7b7bb878a2d9381a977d7ea4a7d2aa7f0595f02905e8c5808c19a1bc111d79
nariq_dir=/data/models/qwen3.8/nari-labs/Qwen3.8-27B-NariQ-GGUF
# LAUNCH-FLAGS: chosen from sweep/ (see REPORT.md); override with NQ_FLAGS.
read -r -a nq_flags <<< "${NQ_FLAGS:--t 1 -tb 8 -ub 2048 -b 2048 --cache-type-k q8_0 --cache-type-v q8_0 --spec-type ngram-mod}"

gsq_dir=/data/models/qwen3.8/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF

ARMS=(nariq_3p0 nariq_2p5 ctl_llamacpp_gsq)
(($#)) && ARMS=("$@")

active_container=
profile_applied=false

cleanup() {
  [[ -z $active_container ]] || docker rm -f "$active_container" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

die() { echo "precheck: $*" >&2; exit 1; }

# ---- prechecks (no side effects) -------------------------------------------
[[ -f $plan ]] || die "missing $plan (run ./build-plan.py)"
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || die "GPU has compute processes; free it first"
systemctl is-active --quiet vram-fan-curve.service || die "vram-fan-curve inactive"
[[ ! -e /run/gpu-thermal-trip ]] || die "/run/gpu-thermal-trip present"
for p in 18181 18184; do
  ! curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$p/" || die "port $p already in use"
done
mkdir -p "$results"

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
    nariq_3p0|nariq_2p5)
      local bpw=3.0 ctx=262144; [[ $arm == nariq_2p5 ]] && bpw=2.5
      port=18184 health=/health ready_attempts=120 max_ctx=$ctx model=nariq-$bpw-qwen3.8-27b
      docker run -d --name "$active_container" --gpus all -p 127.0.0.1:$port:$port \
        --memory 32g --memory-swap 32g --cpus 8 -v "$nariq_dir:/models:ro" "$nariq_image" \
        --model /models/Qwen3.8-27B-NariQ-${bpw}bpw.gguf --mmproj /models/Qwen3.8-27B-NariQ-mmproj-BF16.gguf \
        --ctx-size $max_ctx --parallel 1 --n-gpu-layers 999 --flash-attn on --fit off --jinja \
        --min-p 0 "${nq_flags[@]}" --host 0.0.0.0 --port $port --alias $model ;;
    ctl_llamacpp_gsq)
      port=18181 health=/health ready_attempts=120 max_ctx=200704 model=gsq-qwen3.8-27b
      docker run -d --name "$active_container" --gpus all -p 127.0.0.1:$port:$port \
        --memory 32g --memory-swap 32g --cpus 8 -v "$gsq_dir:/models:ro" "$llama_image" \
        --model /models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --mmproj /models/mmproj-Qwen3.8-27B-BF16.gguf \
        --ctx-size $max_ctx --parallel 1 --n-gpu-layers 999 --flash-attn on \
        --cache-type-k q8_0 --cache-type-v q8_0 --fit off --jinja \
        --spec-type draft-mtp --spec-draft-n-max 2 \
        --host 0.0.0.0 --port $port --alias $model --threads 10 ;;
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
  printf "%s\n" "${nq_flags[*]}" > "$out/nq-flags.txt"
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
