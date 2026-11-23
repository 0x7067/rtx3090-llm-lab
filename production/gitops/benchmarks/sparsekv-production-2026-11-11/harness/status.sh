#!/usr/bin/env bash
set -eu
arm_output=${1:?Usage: status.sh ARM_OUTPUT_DIR}
state="$arm_output/phases.tsv"
if [[ ! -f "$state" ]]; then
  printf 'No phases recorded at %s\n' "$state"
  exit 1
fi
cat "$state"
for report in "$arm_output"/*-summary.json; do
  if [[ -f "$report" ]]; then
    jq -c '{file: input_filename, passed, total, sizes, parallel_e2e_output_tps}' "$report"
  fi
done
