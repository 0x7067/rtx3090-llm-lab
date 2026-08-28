#!/usr/bin/env bash
# How much VRAM do concurrent image requests actually cost per slot?
#
# The deployment manifest records +416 MiB under a 3000x2000 image, measured at
# --parallel 1 and explicitly called "the worst case ONE slot can produce".
# Adding --kv-unified --parallel N means N images can be in flight at once, and
# nothing has measured whether that peak multiplies. At 184320/V=q8_0 the card
# has about 1,172 MiB spare, so if it does multiply, four slots do not fit.
#
# For each slot count this loads the shipped config, measures VRAM at rest, then
# fires N concurrent image requests while sampling nvidia-smi, and reports the
# peak. A row that dies is an OOM and is reported as such.
#
# Needs the whole card. Suspends Flux before scaling and restores on exit.
#
# Usage: sweep_vision_slots.sh [slots ...]   (default: 1 2 4)
set -uo pipefail
cd "$(dirname "$0")"
MODELS=/data/buttercup_6tb/k3s/vllm-trial/models
IMG=ghcr.io/ggml-org/llama.cpp@sha256:851b3b87f89bda98f2ad416e71ab91b6e88be1807502a963937f1d21f3b8555d
PORT=8097
NAME=vslots
CTX=184320
OUT=${OUT:-results_vision_slots.json}
SLOTS=("$@")
[ "$#" -eq 0 ] && SLOTS=(1 2 4)

restore() {
  # The sampler outlives the script if it is not killed here: it is reparented
  # and keeps polling nvidia-smi every 0.3s forever.
  kill "${sampler:-0}" 2>/dev/null || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  rm -f /tmp/vision_probe.png /tmp/vision_probe.b64 /tmp/vbody_*.json /tmp/vcurl_*.err
  echo "== restoring deployment"
  ./restore_qwen.sh || echo "RESTORE FAILED - check kubectl -n apps get pods" >&2
}
trap restore EXIT

echo "== building the 3000x2000 probe image (the manifest's worst case)"
uv run --with pillow python - <<'PY'
from PIL import Image, ImageDraw
import random
# Photographic-ish noise, not flat colour: a solid image can compress or encode
# to something much cheaper than a real screenshot.
img = Image.new("RGB", (3000, 2000))
px = img.load()
random.seed(7)
for y in range(0, 2000, 4):
    for x in range(0, 3000, 4):
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for dy in range(4):
            for dx in range(4):
                px[x + dx, y + dy] = c
d = ImageDraw.Draw(img)
for i in range(40):
    d.text((50, 40 * i + 10), f"Traceback (most recent call last): line {i}", fill=(255, 255, 255))
img.save("/tmp/vision_probe.png")
PY
base64 -w0 /tmp/vision_probe.png > /tmp/vision_probe.b64
echo "   probe: $(stat -c%s /tmp/vision_probe.png) bytes, $(stat -c%s /tmp/vision_probe.b64) base64"
# Bodies go to files: the base64 alone exceeds ARG_MAX, so an inline -d never
# reaches curl. One body per slot index so the prompts stay distinguishable.
for i in $(seq 0 8); do
  python3 - "$i" <<'PY' > "/tmp/vbody_$i.json"
import json, sys
i = sys.argv[1]
b64 = open("/tmp/vision_probe.b64").read().strip()
json.dump({"model": "t", "max_tokens": 64, "temperature": 0,
           "messages": [{"role": "user", "content": [
               {"type": "text", "text": f"Read any text in this image (variant {i})."},
               {"type": "image_url",
                "image_url": {"url": "data:image/png;base64," + b64}}]}]},
          sys.stdout)
PY
done

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

# Sample VRAM continuously into a file; the max is the peak.
#
# The subshell MUST redirect its own stdout. Backgrounding it inside $( ) still
# hands it the command-substitution pipe, and $( ) reads until every writer
# closes that pipe -- so an unredirected sampler loop holds it open forever and
# the caller blocks before the first request is ever sent. That looks exactly
# like a hung model: container up, GPU flat at its load figure, no request in
# the server log.
start_sampler() { : > /tmp/vram_samples;
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      >> /tmp/vram_samples; sleep 0.3; done ) >/dev/null 2>&1 & echo $!; }

one_image_request() {  # index
  curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "@/tmp/vbody_$1.json" \
    -o "/tmp/vresp_$1.json" 2>"/tmp/vcurl_$1.err"
}

echo "[" > "$OUT"
first=1
for np in "${SLOTS[@]}"; do
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  sleep 4
  docker run -d --name "$NAME" --gpus all --user 1000:1000 \
    -p "127.0.0.1:$PORT:8080" -v "$MODELS:/models:ro" "$IMG" --server \
    -m /models/Tiel-Coder-35B-A3B-UD-Q4_K_S.gguf \
    --mmproj /models/Tiel-mmproj-BF16.gguf -ngl 99 \
    -c "$CTX" --parallel "$np" --kv-unified -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --metrics --alias t --host 0.0.0.0 --port 8080 >/dev/null 2>&1
  ready=no
  for _ in $(seq 1 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/health" || true)" = "200" ] && { ready=yes; break; }
    [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo false)" = "false" ] && break
    sleep 5
  done
  if [ "$ready" != yes ]; then
    echo "np=$np -> DIED AT LOAD (OOM)"
    row=$(printf '{"parallel":%s,"status":"oom_at_load"}' "$np")
  else
    load=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    slot=$(docker logs "$NAME" 2>&1 | grep -o 'n_ctx_slot = [0-9]*' | tail -1 | awk '{print $3}')

    sampler=$(start_sampler)
    one_image_request 0
    kill "$sampler" 2>/dev/null
    peak1=$(sort -n /tmp/vram_samples | tail -1)

    sampler=$(start_sampler)
    reqs=()
    for i in $(seq 1 "$np"); do one_image_request "$i" & reqs+=($!); done
    wait "${reqs[@]}" 2>/dev/null
    kill "$sampler" 2>/dev/null
    peakn=$(sort -n /tmp/vram_samples | tail -1)

    ok=0
    for i in $(seq 1 "$np"); do
      python3 -c "import json,sys; d=json.load(open('/tmp/vresp_$i.json')); sys.exit(0 if d.get('choices') else 1)" 2>/dev/null && ok=$((ok+1))
    done
    alive=$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo false)
    echo "np=$np -> ctx_slot=$slot load=${load} peak1img=${peak1} peak${np}img=${peakn} MiB  ($ok/$np replies, container alive=$alive)"
    row=$(printf '{"parallel":%s,"status":"ok","n_ctx_slot":%s,"load_mib":%s,"peak_1_image_mib":%s,"peak_n_images_mib":%s,"replies_ok":%s,"replies_expected":%s,"alive_after":"%s"}' \
      "$np" "${slot:-0}" "$load" "$peak1" "$peakn" "$ok" "$np" "$alive")
    rm -f /tmp/vresp_*.json
  fi
  [ $first -eq 1 ] && first=0 || echo "," >> "$OUT"
  printf '%s' "$row" >> "$OUT"
done
echo "" >> "$OUT"; echo "]" >> "$OUT"
echo "wrote $OUT"
echo "VISION_SLOTS_DONE"
