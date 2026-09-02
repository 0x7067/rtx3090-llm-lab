#!/usr/bin/env bash
# Run the full battery for one engine arm and append to results.jsonl.
#
# Usage: run-arm.sh <tag> <base-url> [api-key]
#   tag       arm name, e.g. vllm / llama-cpp / sglang
#   base-url  OpenAI-compatible base URL, e.g. http://127.0.0.1:18080/v1
#   api-key   optional bearer token
#
# Order: warm-up -> quality -> decode -> sustained -> prefill(30000) ->
#        session(8 turns, no preamble) -> session(6 turns, 50000-token preamble) ->
#        concurrent(4) -> session(20 turns, 20000-token preamble). Model name
#        is fixed to qwen3.8-27b (override via QWEN_MODEL env var if needed).
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <tag> <base-url> [api-key]" >&2
  exit 1
fi

TAG="$1"
BASE_URL="$2"
API_KEY="${3:-}"
MODEL="${QWEN_MODEL:-qwen3.8-27b}"

cd "$(dirname "$0")"

# Pin reasoning effort across engines: vLLM's template default is xhigh, the llama.cpp config's default is medium.
ARGS=(--base-url "$BASE_URL" --model "$MODEL" --tag "$TAG" --out results.jsonl --reasoning "${REASONING:-medium}")
if [ -n "$API_KEY" ]; then
  ARGS+=(--api-key "$API_KEY")
fi

echo "== [$TAG] warm-up request =="
python3 bench.py decode "${ARGS[@]}" --n 1 --out /dev/null

echo "== [$TAG] quality =="
python3 bench.py quality "${ARGS[@]}"

echo "== [$TAG] decode =="
python3 bench.py decode "${ARGS[@]}"

echo "== [$TAG] sustained (long generation, windowed decode tok/s) =="
python3 bench.py sustained "${ARGS[@]}"

echo "== [$TAG] prefill (30000 tokens) =="
python3 bench.py prefill "${ARGS[@]}" --tokens 30000

echo "== [$TAG] session (8 turns, no preamble) =="
python3 bench.py session "${ARGS[@]}" --turns 8 --preamble-tokens 0

echo "== [$TAG] session (6 turns, 50000-token preamble) =="
python3 bench.py session "${ARGS[@]}" --turns 6 --preamble-tokens 50000

echo "== [$TAG] concurrent (n=4) =="
python3 bench.py concurrent "${ARGS[@]}" --n 4

echo "== [$TAG] session (20 turns, 20000-token preamble) =="
python3 bench.py session "${ARGS[@]}" --turns 20 --preamble-tokens 20000

echo "== [$TAG] done: results appended to results.jsonl =="
