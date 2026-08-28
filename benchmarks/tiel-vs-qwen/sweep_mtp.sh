#!/usr/bin/env bash
# Size an MTP build and sweep llama.cpp's two speculative-decoding knobs.
#
# The MTP head adds ~0.9 GB of weights, which comes straight out of the KV
# budget, so the context that fits has to be found before the speed knobs mean
# anything. Both happen in one GPU window: repeated 21 GB reloads from the model
# volume are the part that correlated with a host crash on 2026-08-26, so the
# fewer windows the better.
#
# Phase 1 loads each candidate context and records VRAM, then picks the largest
# that keeps a headroom margin for the vision peak.
# Phase 2 holds that context and sweeps --spec-draft-n-max x --spec-draft-p-min
# against --spec-type none, reporting decode tok/s. The model card is explicit
# that the winning combination differs per machine and that tok/s, not
# acceptance rate, is what to judge on.
#
# Runs at --parallel 2 --no-kv-unified: partitioned slots so one oversized
# request cannot take another session down with it.
#
# Usage: sweep_mtp.sh MODEL_FILE [ctx1,ctx2,...] [headroom_mib]
set -uo pipefail
cd "$(dirname "$0")"
MODELS=/data/buttercup_6tb/k3s/vllm-trial/models
IMG=ghcr.io/ggml-org/llama.cpp@sha256:851b3b87f89bda98f2ad416e71ab91b6e88be1807502a963937f1d21f3b8555d
PORT=8098
NAME=mtpsweep
MODEL_FILE=${1:?usage: sweep_mtp.sh MODEL_FILE [ctx list] [headroom_mib]}
IFS=',' read -ra CTXS <<< "${2:-131072,163840,184320,196608}"
# A 3000x2000 image costs 416 MiB on top of load, measured 2026-08-26, and it
# does not multiply across slots. Keep that plus a little slack.
HEADROOM=${3:-600}
OUT=${OUT:-results_mtp.json}
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)

restore() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "== restoring deployment"
  ./restore_qwen.sh || echo "RESTORE FAILED - check kubectl -n apps get pods" >&2
}
trap restore EXIT

start() {  # ctx spec_type n_max p_min  -> echoes "ok" or "died"
  local ctx=$1 spec=$2 nmax=$3 pmin=$4
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  sleep 4
  local args=(--server -m "/models/$MODEL_FILE"
    --mmproj /models/Tiel-mmproj-BF16.gguf -ngl 99
    -c "$ctx" --parallel 2 --no-kv-unified -fa on
    --cache-type-k q8_0 --cache-type-v q8_0
    --jinja --metrics --alias t --host 0.0.0.0 --port 8080)
  [ "$spec" != none ] && args+=(--spec-type "$spec" --spec-draft-n-max "$nmax" --spec-draft-p-min "$pmin")
  docker run -d --name "$NAME" --gpus all --user 1000:1000 \
    -p "127.0.0.1:$PORT:8080" -v "$MODELS:/models:ro" "$IMG" "${args[@]}" >/dev/null 2>&1
  for _ in $(seq 1 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/health" || true)" = 200 ] && { echo ok; return; }
    [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo false)" = false ] && { echo died; return; }
    sleep 5
  done
  echo died
}

decode_rate() {  # -> "tok/s"
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"t","messages":[{"role":"user","content":"Write a Python class that parses nginx access logs with a regex, handles malformed lines, and explain each design choice."}],"max_tokens":400,"temperature":0}' \
  | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); print(round(d['timings']['predicted_per_second'],1))
except Exception: print(0)
"
}

echo "== suspending flux (must precede the scale-down)"
flux suspend kustomization apps
echo "== scaling apps/llama to 0 and waiting for VRAM"
kubectl -n apps scale deploy llama --replicas=0
kubectl -n apps wait --for=delete pod -l app.kubernetes.io/name=llama --timeout=300s || true
for _ in $(seq 1 60); do
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 500 ] && break
  sleep 3
done

echo
echo "== phase 1: which context fits (card total ${TOTAL} MiB, reserving ${HEADROOM} for vision)"
best_ctx=0
rows_ctx=""
for ctx in "${CTXS[@]}"; do
  if [ "$(start "$ctx" none 0 0)" != ok ]; then
    echo "   ctx=$ctx -> DIED AT LOAD (OOM)"
    rows_ctx="$rows_ctx{\"ctx\":$ctx,\"status\":\"oom\"},"
    continue
  fi
  load=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  slot=$(docker logs "$NAME" 2>&1 | grep -o 'n_ctx_slot = [0-9]*' | tail -1 | awk '{print $3}')
  free=$((TOTAL - load))
  verdict=too-tight
  if [ "$free" -ge "$HEADROOM" ]; then verdict=fits; best_ctx=$ctx; fi
  echo "   ctx=$ctx -> n_ctx_slot=$slot load=${load} MiB free=${free} ($verdict)"
  rows_ctx="$rows_ctx{\"ctx\":$ctx,\"status\":\"ok\",\"n_ctx_slot\":${slot:-0},\"load_mib\":$load,\"free_mib\":$free,\"verdict\":\"$verdict\"},"
done

if [ "$best_ctx" -eq 0 ]; then
  echo "== no candidate context left ${HEADROOM} MiB free; stopping before the sweep"
  printf '{"model":"%s","contexts":[%s],"spec":[]}\n' "$MODEL_FILE" "${rows_ctx%,}" > "$OUT"
  exit 1
fi

echo
echo "== phase 2: MTP sweep at ctx=$best_ctx"
rows_spec=""
for combo in "none 0 0" "draft-mtp 1 0.0" "draft-mtp 3 0.0" "draft-mtp 1 0.8" "draft-mtp 3 0.8" "draft-mtp 8 0.0"; do
  read -r spec nmax pmin <<< "$combo"
  if [ "$(start "$best_ctx" "$spec" "$nmax" "$pmin")" != ok ]; then
    echo "   $spec n_max=$nmax p_min=$pmin -> DIED AT LOAD"
    rows_spec="$rows_spec{\"spec\":\"$spec\",\"n_max\":$nmax,\"p_min\":$pmin,\"status\":\"died\"},"
    continue
  fi
  load=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  r1=$(decode_rate); r2=$(decode_rate)
  echo "   $spec n_max=$nmax p_min=$pmin -> ${r1} and ${r2} tok/s, load ${load} MiB"
  rows_spec="$rows_spec{\"spec\":\"$spec\",\"n_max\":$nmax,\"p_min\":$pmin,\"status\":\"ok\",\"load_mib\":$load,\"tok_s\":[$r1,$r2]},"
done

printf '{"model":"%s","parallel":2,"kv":"partitioned","chosen_ctx":%s,"contexts":[%s],"spec":[%s]}\n' \
  "$MODEL_FILE" "$best_ctx" "${rows_ctx%,}" "${rows_spec%,}" > "$OUT"
echo
echo "wrote $OUT"
echo "MTP_SWEEP_DONE"
