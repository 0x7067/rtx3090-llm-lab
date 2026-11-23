#!/bin/bash
# The C1 row of bench/run_benchmarks.sh, repeated: 8 realistic prompts, 1,024 tokens each,
# one at a time. Prints tokens/step and ms/step per repeat, which is what to compare when
# two configurations look different (greedy e2e moves between sessions; tokens/step does not).
#
#   bash bench/real_rep.sh <tag> [reps] [temperature]     # temperature 0 = greedy
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"; cd "$REPO"
export PATH="$REPO/venv/bin:$PATH"
export OPENAI_API_KEY=${VLLM_API_KEY:-$(cat "$REPO/api_key.txt" 2>/dev/null)}
M=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}; TAG=$1; N=${2:-3}; T=${3:-}
B="venv/bin/vllm bench serve --host 127.0.0.1 --port 18020 --model $M --served-model-name qwen3.8-27b"
metrics() { curl -s http://127.0.0.1:18020/metrics -H "Authorization: Bearer $OPENAI_API_KEY"; }
snap() { metrics | grep -E "^vllm:spec_decode_num_(drafts|accepted_tokens)_total" | grep -v created | awk '{print $NF}' | tr "\n" " "; }
for i in $(seq 1 $N); do
  S0=$(snap)
  [ -n "$T" ] && TA="--temperature $T" || TA=""
  $B --dataset-name custom --dataset-path $HERE/prompts_real.jsonl --custom-output-len 1024 --num-prompts 8 --max-concurrency 1 $TA > /tmp/rr_$TAG_$i.log 2>&1
  S1=$(snap)
  OUT=$(awk '/Total generated tokens/ {print $4}' /tmp/rr_$TAG_$i.log); DUR=$(awk '/Benchmark duration/ {print $4}' /tmp/rr_$TAG_$i.log); E2E=$(awk '/Output token throughput/ {print $5}' /tmp/rr_$TAG_$i.log); TPOT=$(awk '/Mean TPOT/ {print $4}' /tmp/rr_$TAG_$i.log)
  python3 - "$S0" "$S1" "$TAG" "$i" "$OUT" "$DUR" "$E2E" "$TPOT" <<PY
import sys
a=[float(x) for x in sys.argv[1].split()]; b=[float(x) for x in sys.argv[2].split()]
d=[y-x for x,y in zip(a,b)]  # drafts, accepted
steps=d[0]; acc=d[1]
print(f"REP {sys.argv[3]} #{sys.argv[4]} out={sys.argv[5]} dur={sys.argv[6]}s e2e={sys.argv[7]} tok/s decode={1000/float(sys.argv[8]):.1f} tok/s | steps={steps:.0f} tok/step={1+acc/steps:.2f} ms/step={1000*float(sys.argv[6])/steps:.1f}")
PY
done
