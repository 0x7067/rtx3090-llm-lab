#!/usr/bin/env bash
# Exit 0 once qwen3.8 has had <=1 chat request in the trailing 5 minutes
# (allowance for the 15-min cache canary). Exit 2 after 6h.
for i in $(seq 1 360); do
  n=$(kubectl -n apps logs deploy/llama --since=300s 2>/dev/null | grep -c 'POST /v1/chat/completions' || echo 99)
  if [ "$n" -le 1 ]; then
    echo "idle: $n requests in last 5m"
    exit 0
  fi
  echo "$(date +%H:%M:%S) busy: $n requests in last 5m"
  sleep 60
done
exit 2
