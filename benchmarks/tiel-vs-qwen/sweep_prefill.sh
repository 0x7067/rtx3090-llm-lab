#!/usr/bin/env bash
# Isolate the cause of the prefill collapse.
#
# The shipped config prefills a ~6.8k prompt at 188 tok/s. The same model and
# engine did 2805 tok/s on the 2026-08-25 bench shape. Four things differ
# between them, so this varies them one at a time:
#
#   A  262144  K=q8_0 V=q4_0  vision     the shipped config (expect ~188)
#   B  262144  K=q8_0 V=q4_0  no vision  isolates the projector
#   C   65536  K=q8_0 V=q4_0  vision     isolates context depth
#   D   65536  f16   f16      no vision  closest to the bench shape (expect ~2805)
#   E  131072  K=q8_0 V=q8_0  vision     the candidate replacement config
#
# Reading it: if C is fast, depth is the cause. If C is slow and D is fast, KV
# quantization is. If D is also slow, nothing about the 2805 measurement
# reproduces and the difference is elsewhere in the build.
#
# Every row runs --parallel 1 and is measured with bench_speed.py, the same
# harness that produced the 188 and 1269 tok/s figures in REPORT.md.
#
# Needs the whole card. Suspends Flux before scaling and restores on exit.
#
# Usage: sweep_prefill.sh ["TAG CTX K V vision|novision" ...]
#
# With no arguments it runs the five rows above. Pass rows to measure a
# different set, for example a depth ladder at the KV setting this identified:
#
#   OUT=results_depth.json ./sweep_prefill.sh \\
#     "F 163840 q8_0 q8_0 vision" "G 184320 q8_0 q8_0 vision"
#
# Expect roughly 5 minutes per row plus the restore.
set -uo pipefail
cd "$(dirname "$0")"
MODELS=/data/buttercup_6tb/k3s/vllm-trial/models
IMG=ghcr.io/ggml-org/llama.cpp@sha256:851b3b87f89bda98f2ad416e71ab91b6e88be1807502a963937f1d21f3b8555d
PORT=8096
NAME=prefill
OUT=${OUT:-results_prefill_isolation.json}

restore() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "== restoring deployment"
  ./restore_qwen.sh || echo "RESTORE FAILED - check kubectl -n apps get pods" >&2
}
trap restore EXIT

echo "== suspending flux (must precede the scale-down)"
flux suspend kustomization apps
echo "== scaling apps/llama to 0 and waiting for VRAM"
kubectl -n apps scale deploy llama --replicas=0
kubectl -n apps wait --for=delete pod -l app.kubernetes.io/name=llama --timeout=300s || true
for _ in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$u" -lt 500 ] && break
  sleep 3
done

start_cfg() {  # ctx ck cv vision
  local ctx="$1" ck="$2" cv="$3" vis="$4"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  sleep 4
  local mm=()
  [ "$vis" = "vision" ] && mm=(--mmproj /models/Tiel-mmproj-BF16.gguf)
  docker run -d --name "$NAME" --gpus all --user 1000:1000 \
    -p "127.0.0.1:$PORT:8080" -v "$MODELS:/models:ro" "$IMG" --server \
    -m /models/Tiel-Coder-35B-A3B-UD-Q4_K_S.gguf "${mm[@]}" -ngl 99 \
    -c "$ctx" --parallel 1 -fa on \
    --cache-type-k "$ck" --cache-type-v "$cv" \
    --jinja --metrics --alias t --host 0.0.0.0 --port 8080 >/dev/null 2>&1
  for _ in $(seq 1 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/health" || true)" = "200" ] && return 0
    [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo false)" = "false" ] && return 1
    sleep 5
  done
  return 1
}

echo "[" > "$OUT"
first=1
#        label ctx     k     v     vision
CFGS=("A 262144 q8_0 q4_0 vision"
      "B 262144 q8_0 q4_0 novision"
      "C 65536  q8_0 q4_0 vision"
      "D 65536  f16  f16  novision"
      "E 131072 q8_0 q8_0 vision")
[ "$#" -gt 0 ] && CFGS=("$@")
for cfg in "${CFGS[@]}"; do
  read -r tag ctx ck cv vis <<< "$cfg"
  label="$tag ctx=$ctx K=$ck V=$cv $vis"
  if ! start_cfg "$ctx" "$ck" "$cv" "$vis"; then
    echo "$label -> DIED/TIMEOUT (likely OOM)"
    row=$(printf '{"tag":"%s","ctx":%s,"cache_k":"%s","cache_v":"%s","vision":"%s","status":"failed"}' \
      "$tag" "$ctx" "$ck" "$cv" "$vis")
  else
    vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    tmp=$(mktemp)
    uv run bench_speed.py "http://127.0.0.1:$PORT" t "$tmp" >/dev/null 2>&1
    read -r dec pre < <(python3 - "$tmp" <<'PY'
import json, statistics as st, sys
d = json.load(open(sys.argv[1]))
print(round(st.median(r["gen_tok_s"] for r in d["single_stream_decode"]), 1),
      round(st.median(r["prefill_tok_s"] for r in d["prefill"]), 1))
PY
)
    echo "$label -> ${vram}MiB  decode=${dec} tok/s  prefill=${pre} tok/s"
    row=$(printf '{"tag":"%s","ctx":%s,"cache_k":"%s","cache_v":"%s","vision":"%s","status":"ok","vram_mib":%s,"decode_tok_s":%s,"prefill_tok_s":%s}' \
      "$tag" "$ctx" "$ck" "$cv" "$vis" "$vram" "$dec" "$pre")
    rm -f "$tmp"
  fi
  [ $first -eq 1 ] && first=0 || echo "," >> "$OUT"
  printf '%s' "$row" >> "$OUT"
done
echo "" >> "$OUT"; echo "]" >> "$OUT"
echo "wrote $OUT"
echo "PREFILL_SWEEP_DONE"
