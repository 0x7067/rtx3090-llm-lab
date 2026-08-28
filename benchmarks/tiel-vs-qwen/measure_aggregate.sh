#!/usr/bin/env bash
# Single-stream and N-way aggregate decode throughput against a live endpoint.
#
# Unlike sweep_parallel.sh this starts no server: it measures whatever is
# already serving, so a before/after around a deployment change needs no
# outage. Point it at the cluster IP and it measures production as configured.
#
# Every rate is reported as "rate/completions". A response that is not a
# completion contributes zero tokens but full wall time, so a rate without its
# completion count is unreadable.
#
# Usage: measure_aggregate.sh BASE_URL MODEL [concurrency] [max_tokens]
set -uo pipefail
BASE=${1:?usage: measure_aggregate.sh BASE_URL MODEL [concurrency] [max_tokens]}
MODEL=${2:?}
CONC=${3:-4}
MAXTOK=${4:-400}
PROMPT="Write a Python function to parse an nginx log line with a regex, then explain each capture group"

measure() {  # concurrency
  local n="$1" pids=() i dir t0 t1
  dir=$(mktemp -d)
  t0=$(date +%s.%N)
  for i in $(seq 1 "$n"); do
    curl -s -m 600 "$BASE/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT (variant $i)\"}],\"max_tokens\":$MAXTOK,\"temperature\":0}" \
      > "$dir/$i.json" 2>/dev/null &
    pids+=($!)
  done
  wait "${pids[@]}" 2>/dev/null
  t1=$(date +%s.%N)
  python3 - "$dir" "$t0" "$t1" <<'PY'
import glob, json, shutil, sys
d, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
tot = ok = 0
for f in glob.glob(d + "/*.json"):
    try:
        tot += json.load(open(f))["usage"]["completion_tokens"]
        ok += 1
    except Exception:
        pass
print(f"{round(tot / (t1 - t0), 1)}/{ok}")
shutil.rmtree(d, ignore_errors=True)
PY
}

echo "endpoint: $BASE  model: $MODEL  max_tokens: $MAXTOK"
echo "single stream:      $(measure 1) tok/s per completion"
echo "$CONC concurrent:        $(measure "$CONC") tok/s aggregate per completion"
