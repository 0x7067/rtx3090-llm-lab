#!/usr/bin/env bash
# Does an IDLE slot give its KV back to the unified pool?
#
# test_pool_limit.sh showed that slots which are all ACTIVE at once fail
# together with HTTP 500 once their live tokens exceed the pool. That is the
# loud case. This measures the quiet one: whether a session that has finished
# its turn keeps occupying the pool while another session runs.
#
# llama.cpp defaults --cache-idle-slots to enabled with --cache-ram 8192, which
# is documented as "save idle slots to the prompt cache on new task, and clear
# them when using unified KV". Neither flag is set explicitly on the
# deployment, so both defaults apply. This checks the behaviour rather than
# trusting the help text, by sampling /slots through the whole sequence.
#
# Usage: test_idle_eviction.sh [words_each]
set -uo pipefail
BASE=${BASE:-http://llama.apps.svc.cluster.local:8080}
MODEL=${MODEL:-tiel-coder-35b-a3b}
WORDS=${1:-110000}
SAMPLES=/tmp/evict_slots.log

cleanup() { kill "${poller:-0}" 2>/dev/null; }
trap cleanup EXIT

echo "== building two prompts of ~$WORDS words each"
python3 - "$WORDS" "$MODEL" <<'PY'
import json, random, sys
words_n, model = int(sys.argv[1]), sys.argv[2]
vocab = ['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa']
random.seed(41)
for i in (1, 2):
    # Distinct text per request so prefix reuse cannot serve the second one.
    text = ' '.join(random.choice(vocab) for _ in range(words_n))
    json.dump({"model": model, "max_tokens": 32, "temperature": 0,
               "messages": [{"role": "user", "content": f"Doc {i}. Summarise briefly.\n\n{text}"}]},
              open(f"/tmp/evict_{i}.json", "w"))
PY

# Sample /slots continuously. The subshell redirects its own stdout so the
# command substitution that captures its pid is not held open by it.
: > "$SAMPLES"
( while true; do
    printf '%s ' "$(date -u +%H:%M:%S)" >> "$SAMPLES"
    curl -s -m 3 "$BASE/slots" 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(' '.join(f\"slot{s['id']}={s.get('n_prompt_tokens',0)}{'*' if s.get('is_processing') else ''}\" for s in d))
except Exception:
    print('unreadable')
" >> "$SAMPLES" 2>/dev/null
    sleep 1
  done ) >/dev/null 2>&1 &
poller=$!

echo "== A: deep request, then leaves its slot idle"
codeA=$(curl -s -m 900 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  --data-binary @/tmp/evict_1.json -o /tmp/evict_r1.json -w '%{http_code}')
tokA=$(python3 -c "import json;d=json.load(open('/tmp/evict_r1.json'));print(d.get('usage',{}).get('prompt_tokens','?'))")
echo "   A: HTTP $codeA, prompt_tokens $tokA"
echo "   /slots right after A:"; sed -n '$p' "$SAMPLES"

echo "== B: second deep request. Sum of A and B far exceeds the pool."
codeB=$(curl -s -m 900 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  --data-binary @/tmp/evict_2.json -o /tmp/evict_r2.json -w '%{http_code}')
tokB=$(python3 -c "import json;d=json.load(open('/tmp/evict_r2.json'));print(d.get('usage',{}).get('prompt_tokens','?'))")
echo "   B: HTTP $codeB, prompt_tokens $tokB"

kill "$poller" 2>/dev/null
echo
echo "== verdict"
echo "   A $tokA + B $tokB = $((tokA + tokB)) tokens against a 184320 pool"
[ "$codeA" = "200" ] && [ "$codeB" = "200" ] \
  && echo "   both succeeded: an idle slot does NOT hold the pool" \
  || echo "   a request failed: idle slots DO hold the pool"
echo
echo "== /slots timeline (n_prompt_tokens per slot, * = processing)"
uniq -f1 "$SAMPLES" | tail -25
