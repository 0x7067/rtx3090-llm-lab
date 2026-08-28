#!/usr/bin/env bash
# Does the unified KV pool actually bind on the SUM of live tokens?
#
# PARALLELIZATION.md and the deployment manifest both claim it does: with
# --kv-unified the constraint is said to become "total live tokens across all
# sequences must fit the pool" rather than ctx/N per caller. Nobody measured it.
#
# A first attempt sent 4 x 50k against a 184,320 pool and all four completed
# intact -- but their prefills serialised about 29s apart, so they may never
# have been resident together. This forces overlap: each request asks for a
# long generation, so a slot keeps holding its KV long after prefill, and by
# the time the last one prefills the first three are still decoding.
#
# Usage: test_pool_limit.sh [tokens_each] [concurrency] [max_tokens]
set -uo pipefail
BASE=${BASE:-http://llama.apps.svc.cluster.local:8080}
MODEL=${MODEL:-tiel-coder-35b-a3b}
EACH=${1:-60000}
CONC=${2:-4}
GEN=${3:-600}

echo "== building $CONC prompts of ~$EACH tokens each"
python3 - "$EACH" "$CONC" "$GEN" "$MODEL" <<'PY'
import json, random, sys
each, conc, gen, model = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
words = ['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa']
random.seed(31)
for i in range(1, conc + 1):
    # A distinct body per request so prefix caching cannot collapse them into one.
    text = ' '.join(random.choice(words) for _ in range(each))
    json.dump({"model": model, "max_tokens": gen, "temperature": 0,
               "messages": [{"role": "user", "content":
                   f"Session {i}. Summarise the following list, then keep writing "
                   f"about it at length until you are told to stop.\n\n{text}"}]},
              open(f"/tmp/pool_{i}.json", "w"))
print(f"   wrote {conc} bodies, {gen} max_tokens each")
PY

since=$(date -u -d '5 seconds ago' +%Y-%m-%dT%H:%M:%SZ)
echo "== firing $CONC concurrent requests"
pids=()
for i in $(seq 1 "$CONC"); do
  curl -s -m 1200 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-binary "@/tmp/pool_$i.json" -o "/tmp/pool_resp_$i.json" -w "%{http_code}" \
    > "/tmp/pool_code_$i" 2>/dev/null &
  pids+=($!)
done
wait "${pids[@]}" 2>/dev/null
echo "== all requests returned"
echo

python3 - "$CONC" <<'PY'
import json, sys
conc = int(sys.argv[1])
total = 0
for i in range(1, conc + 1):
    code = open(f"/tmp/pool_code_{i}").read().strip()
    try:
        d = json.load(open(f"/tmp/pool_resp_{i}.json"))
    except Exception as e:
        print(f"session {i}: HTTP {code}, unreadable body ({e})"); continue
    if "choices" not in d:
        print(f"session {i}: HTTP {code} ERROR -> {json.dumps(d)[:300]}"); continue
    u = d["usage"]
    total += u["prompt_tokens"] + u["completion_tokens"]
    print(f"session {i}: HTTP {code} prompt={u['prompt_tokens']} "
          f"completion={u['completion_tokens']} finish={d['choices'][0]['finish_reason']}")
print(f"\ntotal live tokens if all resident at once: {total}")
PY

echo
echo "== server log: truncation, context shift, eviction, errors"
kubectl -n apps logs -l app.kubernetes.io/name=llama --since-time="$since" --timestamps 2>/dev/null \
  | grep -iE 'truncated = [1-9]|context shift|shifting|erase|cache_prompt|no slot|exceed|error|slot .*(launch|release)' \
  | tail -40
