#!/usr/bin/env bash
# Smoke-test a running SGLang server: wait for /health, then one chat completion
# and one tool call. Prints decode throughput so a 6 tok/s eager result is
# immediately obvious.
#
#   ./smoke.sh                 # against 127.0.0.1:18030
#   PORT=18031 ./smoke.sh
#   TIMEOUT=1800 ./smoke.sh    # first boot JITs Triton kernels; be patient
set -uo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18030}"
BASE="http://${HOST}:${PORT}"
MODEL="${SERVED_NAME:-qwen3.8-27b}"
TIMEOUT="${TIMEOUT:-1800}"
UA="OpenAI File Downloader, XaiImageApiFetch/1.0"
CURL=(curl -sS -A "$UA" -H 'Content-Type: application/json')
FAIL=0

say () { printf '\n=== %s ===\n' "$1"; }

# --------------------------------------------------------------- 1. health ---
say "waiting for ${BASE}/health (up to ${TIMEOUT}s)"
start=$(date +%s)
while :; do
  # curl -w already prints 000 when it cannot connect, so do not add another.
  code=$("${CURL[@]}" -o /dev/null -w '%{http_code}' "${BASE}/health" 2>/dev/null) || true
  code="${code:-000}"
  [[ "$code" == "200" ]] && { echo "healthy after $(( $(date +%s) - start ))s"; break; }
  now=$(date +%s)
  if (( now - start > TIMEOUT )); then
    echo "TIMED OUT after ${TIMEOUT}s (last HTTP $code)." >&2
    echo "Check the server log for these, in order of likelihood:" >&2
    echo "  'state cache is too small to serve any requests'  -> sizing, see NOTES.md" >&2
    echo "  'Capture target decode CUDA graph begin' + 0% GPU -> graph hang, rerun with EAGER=1" >&2
    echo "  'no GPU memory for the KV cache'                  -> lower CTX or raise MFS" >&2
    exit 1
  fi
  sleep 5
done

say "server metadata"
"${CURL[@]}" "${BASE}/get_model_info" 2>/dev/null || echo "(no /get_model_info)"
echo

# ------------------------------------------------------- 2. chat completion ---
say "chat completion"
REQ=$(cat <<JSON
{"model":"${MODEL}",
 "messages":[{"role":"user","content":"Write exactly two sentences about autumn."}],
 "max_tokens":200,"temperature":0.7,"stream":false}
JSON
)
t0=$(date +%s.%N)
RESP=$("${CURL[@]}" -X POST "${BASE}/v1/chat/completions" -d "$REQ" 2>&1)
t1=$(date +%s.%N)

python3 - "$RESP" "$t0" "$t1" <<'PY'
import json, sys
raw, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
dt = t1 - t0
try:
    d = json.loads(raw)
except Exception:
    print("NOT JSON:", raw[:600]); sys.exit(1)
if "error" in d:
    print("SERVER ERROR:", json.dumps(d["error"])[:600]); sys.exit(1)
msg = d["choices"][0]["message"]
txt = (msg.get("content") or "").strip()
think = (msg.get("reasoning_content") or "").strip()
u = d.get("usage", {})
out = u.get("completion_tokens", 0)
print(f"content ({len(txt)} chars): {txt[:300]}")
if think:
    print(f"reasoning_content ({len(think)} chars) -> the qwen3 reasoning parser is live")
print(f"usage: prompt={u.get('prompt_tokens')} completion={out}")
print(f"wall={dt:.1f}s  decode={out/dt:.1f} tok/s" if out and dt > 0 else f"wall={dt:.1f}s")
print()
if out and dt > 0:
    r = out / dt
    if r < 12:
        print(f"NOTE: {r:.1f} tok/s is the eager-decode regime that issue #36048")
        print("      measured (~6 tok/s on SM89). The decode CUDA graph is probably")
        print("      off or ineffective; llama.cpp reaches ~64 tok/s on this class of card.")
    else:
        print(f"{r:.1f} tok/s -- above the eager regime, so a fast path is engaged.")
assert txt, "empty completion"
print("CHAT OK")
PY
[[ $? -eq 0 ]] || FAIL=1

# -------------------------------------------------------------- 3. tool call ---
say "tool call (qwen3_coder parser)"
TREQ=$(cat <<'JSON'
{"model":"__MODEL__",
 "messages":[{"role":"user","content":"What is the weather in Sao Paulo right now? Use the tool."}],
 "tools":[{"type":"function","function":{
    "name":"get_weather",
    "description":"Get the current weather for a city.",
    "parameters":{"type":"object","properties":{
        "city":{"type":"string","description":"City name"},
        "unit":{"type":"string","enum":["celsius","fahrenheit"]}},
      "required":["city"]}}}],
 "tool_choice":"auto","max_tokens":300,"temperature":0}
JSON
)
TREQ=${TREQ/__MODEL__/$MODEL}
TRESP=$("${CURL[@]}" -X POST "${BASE}/v1/chat/completions" -d "$TREQ" 2>&1)

python3 - "$TRESP" <<'PY'
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception:
    print("NOT JSON:", sys.argv[1][:600]); sys.exit(1)
if "error" in d:
    print("SERVER ERROR:", json.dumps(d["error"])[:600]); sys.exit(1)
ch = d["choices"][0]
msg = ch["message"]
tc = msg.get("tool_calls") or []
print("finish_reason:", ch.get("finish_reason"))
if not tc:
    print("NO tool_calls parsed. content was:")
    print((msg.get("content") or "")[:600])
    print("\nIf the raw text contains a <tool_call> block, the parser name is wrong;")
    print("try --tool-call-parser qwen3 or qwen25 instead of qwen3_coder.")
    sys.exit(1)
f = tc[0]["function"]
print("tool name:", f["name"])
print("arguments:", f["arguments"])
args = json.loads(f["arguments"])       # must be valid JSON, not a string blob
assert f["name"] == "get_weather", f"unexpected tool {f['name']}"
assert "city" in args, f"missing required arg city: {args}"
print("TOOL CALL OK")
PY
[[ $? -eq 0 ]] || FAIL=1

say "result"
if (( FAIL )); then echo "SMOKE FAILED"; exit 1; fi
echo "SMOKE PASSED"
