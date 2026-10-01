#!/usr/bin/env bash
set -euo pipefail

task_dir=/data/docker-services/rtx3090-llm-lab/benchmarks/local-llm-reset-2026-09-30

while [[ ! -e /tmp/local-llm-reset-16384-complete ]]; do
  if [[ $(docker inspect thinkingcap-16384-final --format '{{.State.Running}}' 2>/dev/null || true) == true ]]; then
    gputemps --json --once >> "$task_dir/thinkingcap-16384-thermal.jsonl"
  elif [[ $(docker inspect gsq-16384 --format '{{.State.Running}}' 2>/dev/null || true) == true ]]; then
    gputemps --json --once >> "$task_dir/gsq_llama-16384-thermal.jsonl"
  elif [[ $(docker inspect w4-16384 --format '{{.State.Running}}' 2>/dev/null || true) == true ]]; then
    gputemps --json --once >> "$task_dir/w4_vllm-16384-thermal.jsonl"
  elif [[ $(docker inspect orca-16384 --format '{{.State.Running}}' 2>/dev/null || true) == true ]]; then
    gputemps --json --once >> "$task_dir/orca_vllm-16384-thermal.jsonl"
  elif [[ $(docker inspect strata-16384 --format '{{.State.Running}}' 2>/dev/null || true) == true ]]; then
    gputemps --json --once >> "$task_dir/strata_flash_next-16384-thermal.jsonl"
  fi
  sleep 5
done
