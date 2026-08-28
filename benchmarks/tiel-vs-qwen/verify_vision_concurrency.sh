#!/usr/bin/env bash
# Verify on the LIVE deployment that concurrent image requests neither OOM the
# card nor multiply the projector's peak allocation.
#
# sweep_vision_slots.sh measured this on a docker container it started itself.
# This runs the same worst case against whatever is actually serving, and
# captures the server's own slot timestamps so the encodes can be checked for
# overlap -- the mechanism the sweep could not confirm.
#
# Usage: verify_vision_concurrency.sh [BASE_URL] [MODEL] [concurrency]
set -uo pipefail
cd "$(dirname "$0")"
BASE=${1:-http://llama.apps.svc.cluster.local:8080}
MODEL=${2:-tiel-coder-35b-a3b}
CONC=${3:-4}

cleanup() { kill "${sampler:-0}" 2>/dev/null; rm -f /tmp/vv_*.json /tmp/vv_probe.*; }
trap cleanup EXIT

echo "== building the 3000x2000 probe (the manifest's worst case)"
uv run --with pillow python - <<'PY'
from PIL import Image, ImageDraw
import random
img = Image.new("RGB", (3000, 2000)); px = img.load(); random.seed(7)
for y in range(0, 2000, 4):
    for x in range(0, 3000, 4):
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for dy in range(4):
            for dx in range(4):
                px[x + dx, y + dy] = c
d = ImageDraw.Draw(img)
for i in range(40):
    d.text((50, 40 * i + 10), f"Traceback (most recent call last): line {i}", fill=(255, 255, 255))
img.save("/tmp/vv_probe.png")
PY
base64 -w0 /tmp/vv_probe.png > /tmp/vv_probe.b64
# Bodies must be built from a file: the base64 alone exceeds ARG_MAX.
for i in $(seq 1 "$CONC"); do
  MODEL="$MODEL" python3 - "$i" <<'PY' > "/tmp/vv_body_$i.json"
import json, os, sys
b64 = open("/tmp/vv_probe.b64").read().strip()
json.dump({"model": os.environ["MODEL"], "max_tokens": 64, "temperature": 0,
           "messages": [{"role": "user", "content": [
               {"type": "text", "text": f"Read any text in this image (variant {sys.argv[1]})."},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]},
          sys.stdout)
PY
done

# The subshell redirects its own stdout: without that it holds the command
# substitution's pipe open and this call never returns.
start_sampler() { : > /tmp/vv_samples
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      >> /tmp/vv_samples; sleep 0.2; done ) >/dev/null 2>&1 & echo $!; }

rest=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
echo "== at rest: ${rest} MiB of ${total}"

since=$(date -u -d '10 seconds ago' +%Y-%m-%dT%H:%M:%SZ)
sampler=$(start_sampler)
echo "== firing $CONC concurrent image requests at $BASE"
pids=()
for i in $(seq 1 "$CONC"); do
  curl -s -m 900 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-binary "@/tmp/vv_body_$i.json" -o "/tmp/vv_resp_$i.json" &
  pids+=($!)
done
wait "${pids[@]}" 2>/dev/null
kill "$sampler" 2>/dev/null
peak=$(sort -n /tmp/vv_samples | tail -1)

ok=0
for i in $(seq 1 "$CONC"); do
  python3 -c "import json,sys; d=json.load(open('/tmp/vv_resp_$i.json')); sys.exit(0 if d.get('choices') else 1)" 2>/dev/null && ok=$((ok+1))
done

echo
echo "at rest      ${rest} MiB"
echo "peak         ${peak} MiB"
echo "image cost   $((peak - rest)) MiB for $CONC concurrent images"
echo "free at peak $((total - peak)) MiB"
echo "replies      $ok/$CONC"
echo
echo "== slot timestamps (overlap check)"
kubectl -n apps logs -l app.kubernetes.io/name=llama --since-time="$since" --timestamps 2>/dev/null \
  | grep -E 'launch_slot_|slot *release' | tail -$((CONC * 2))
