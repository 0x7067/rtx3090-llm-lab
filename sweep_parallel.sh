#!/usr/bin/env bash
# Measure the slots-by-depth frontier for Tiel on one 24 GB card.
#
# llama.cpp offers two ways to serve more than one request at a time, and they
# differ in how the KV cache is divided:
#
#   --no-kv-unified (implied by an explicit --parallel N): --ctx-size is split
#     statically, so N slots each get ctx/N tokens.
#   --kv-unified: one KV buffer shared by all sequences, isolated by masking.
#     Slots draw from the common pool, so a single request can still reach the
#     full depth while short requests run beside it.
#
# This model is a hybrid: qwen35moe with full_attention_interval=4, so 10 of its
# 40 layers hold a growing KV cache and the other 30 hold a fixed-size recurrent
# state per sequence. That makes depth cheap and slots almost free in theory.
# This script measures whether that holds.
#
# Requires an idle GPU. Run it with the production Deployment scaled to 0 and
# the Flux kustomization suspended.
#
# Usage: sweep_parallel.sh [results.json]

set -uo pipefail
OUT="${1:-results_parallel.json}"
MODELS=${MODELS_DIR:-$PWD/models}
IMG=ghcr.io/ggml-org/llama.cpp@sha256:851b3b87f89bda98f2ad416e71ab91b6e88be1807502a963937f1d21f3b8555d
PORT=8095
NAME=sweep

PROMPT='Write a Python function that parses an nginx access log line with a regex, then explain each capture group.'

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# Start one configuration and wait for health. Echoes nothing; sets globals.
start_cfg() {  # ctx parallel kv_unified
  local ctx="$1" np="$2" kvu="$3"
  cleanup; sleep 4
  local kvflag=""
  [ "$kvu" = "yes" ] && kvflag="--kv-unified"
  docker run -d --name "$NAME" --gpus all --user 1000:1000 \
    -p "127.0.0.1:$PORT:8080" -v "$MODELS:/models:ro" "$IMG" --server \
    -m /models/Tiel-Coder-35B-A3B-UD-Q4_K_S.gguf \
    --mmproj /models/Tiel-mmproj-BF16.gguf -ngl 99 \
    -c "$ctx" --parallel "$np" $kvflag -fa on \
    --cache-type-k q8_0 --cache-type-v q4_0 \
    --jinja --metrics --alias t --host 0.0.0.0 --port 8080 >/dev/null 2>&1
  for _ in $(seq 1 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/health" || true)" = "200" ] && return 0
    [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo false)" = "false" ] && return 1
    sleep 5
  done
  return 1
}

# Aggregate decode throughput across N concurrent requests.
measure() {  # concurrency
  local n="$1" pids=() i
  local dir; dir=$(mktemp -d)
  local t0 t1
  t0=$(date +%s.%N)
  for i in $(seq 1 "$n"); do
    curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"t\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT (variant $i)\"}],\"max_tokens\":400,\"temperature\":0}" \
      > "$dir/$i.json" 2>/dev/null &
    pids+=($!)
  done
  wait "${pids[@]}" 2>/dev/null
  t1=$(date +%s.%N)
  # Reports "rate/completions". A response that is not a completion - an error
  # body, a 503, a truncated write - contributes zero tokens but full wall
  # time, so a rate without the completion count is unreadable. The 2026-08-26
  # run learned this the hard way: the partitioned rows returned 17.3 and 22.9
  # tok/s against a ~114 tok/s single-stream floor, which is only possible if
  # most responses never arrived.
  python3 - "$dir" "$t0" "$t1" <<'PY'
import glob, json, sys
d, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
tot = ok = 0
for f in glob.glob(d + "/*.json"):
    try:
        tot += json.load(open(f))["usage"]["completion_tokens"]
        ok += 1
    except Exception:
        pass
print(f"{round(tot / (t1 - t0), 1)}/{ok}")
PY
  rm -rf "$dir"
}

vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
ctx_slot() { docker logs "$NAME" 2>&1 | grep -o 'n_ctx_slot = [0-9]*' | tail -1 | awk '{print $3}'; }

echo "[" > "$OUT"
first=1
#        ctx     np  kv-unified
for cfg in "262144 1 no" "262144 2 yes" "262144 4 yes" "262144 8 yes" \
           "262144 2 no" "262144 4 no"; do
  read -r ctx np kvu <<< "$cfg"
  label="ctx=$ctx np=$np kv_unified=$kvu"
  if ! start_cfg "$ctx" "$np" "$kvu"; then
    echo "$label -> DIED/TIMEOUT"
    tail_log=$(docker logs --tail 5 "$NAME" 2>&1 | tr '\n' ' ' | tr -d '"')
    row=$(printf '{"ctx":%s,"parallel":%s,"kv_unified":"%s","status":"failed","log":"%s"}' \
      "$ctx" "$np" "$kvu" "${tail_log:0:300}")
  else
    load=$(vram); slot=$(ctx_slot)
    t1=$(measure 1); peak_after_1=$(vram)
    t4=$(measure 4); peak=$(vram)
    echo "$label -> ctx_slot=$slot load=${load}MiB peak=${peak}MiB  1-stream=${t1%%/*} tok/s (${t1#*/}/1 ok)  4-concurrent=${t4%%/*} tok/s (${t4#*/}/4 ok)"
    row=$(printf '{"ctx":%s,"parallel":%s,"kv_unified":"%s","status":"ok","n_ctx_slot":%s,"load_mib":%s,"peak_mib":%s,"decode_1_tok_s":%s,"completions_1":%s,"aggregate_4_tok_s":%s,"completions_4":%s}' \
      "$ctx" "$np" "$kvu" "${slot:-0}" "$load" "$peak" \
      "${t1%%/*}" "${t1#*/}" "${t4%%/*}" "${t4#*/}")
  fi
  [ $first -eq 1 ] && first=0 || echo "," >> "$OUT"
  printf '%s' "$row" >> "$OUT"
done
echo "" >> "$OUT"; echo "]" >> "$OUT"
echo "wrote $OUT"
