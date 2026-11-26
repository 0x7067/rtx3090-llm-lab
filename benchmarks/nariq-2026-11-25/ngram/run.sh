#!/usr/bin/env bash
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
I=ghcr.io/ggml-org/llama.cpp@sha256:c4348afb91499b10fad12f92626c7a8ca8b69d6475d10c793b810cb61237ce00
G=/data/models/qwen3.8/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF
declare -A ARMS=(
 [none]="--spec-type none"
 [mtp2]="--spec-type draft-mtp --spec-draft-n-max 2"
 [ngram]="--spec-type ngram-mod"
 [mtp2+ngram]="--spec-type draft-mtp,ngram-mod --spec-draft-n-max 2"
)
for arm in mtp2 ngram mtp2+ngram none; do
  echo "== $arm $(date -u +%TZ)"; docker rm -f ng-probe >/dev/null 2>&1
  docker run -d --name ng-probe --gpus all -p 127.0.0.1:18185:18185 --memory 32g --memory-swap 32g --cpus 8 -v "$G:/models:ro" "$I" \
    --model /models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --ctx-size 200704 --parallel 1 --n-gpu-layers 999 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 --fit off --jinja --threads 10 ${ARMS[$arm]} --host 0.0.0.0 --port 18185 --alias gsq >/dev/null
  ok=; for i in $(seq 180); do curl -sf localhost:18185/health >/dev/null && { ok=1; break; }; [[ $(docker inspect -f '{{.State.Running}}' ng-probe) == true ]] || break; sleep 2; done
  if [[ -z $ok ]]; then echo "   failed to start"; docker logs ng-probe 2>&1 | grep -E ' E |error' | tail -3; continue; fi
  python3 -I probe.py http://localhost:18185/v1/chat/completions "$arm" "$D/res-$arm.jsonl" | python3 -c 'import sys,json;[print("  ",(r:=json.loads(l))["task"],r["rep"],"tg",r["tg"],"n",r["n"],"acc",r["accepted"],"/",r["draft"],r["finish"]) for l in sys.stdin]'
  docker logs ng-probe 2>&1 | grep -i -E 'draft acceptance|statistics' | tail -2
done
docker rm -f ng-probe >/dev/null; echo "done $(date -u +%TZ)"
