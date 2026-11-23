#!/usr/bin/env bash
set -eu

if (( $# < 4 || $# > 5 )); then
  echo "Usage: $0 BASE MODEL OUTPUT_DIR TRIAL_ID [API_KEY_FILE]" >&2
  exit 1
fi

here=$(cd "$(dirname "$0")" && pwd)
validation="$here/../../../k8s/workloads/apps/llama/optimization/validation/checks.py"
args=(--base "$1" --model "$2" --out "$3")
if (( $# == 5 )); then
  args+=(--api-key-file "$5")
fi

mkdir "$3"
jq -n --arg trial_id "$4" '{trial_id:$trial_id}' > "$3/trial.json"
state="$3/phases.tsv"
status=0
run_phase() {
  local name=$1 duration=$2 rc=0
  shift 2
  printf '%s\t%s\tstart (limit %s)\n' "$(date -Iseconds)" "$name" "$duration" >> "$state"
  timeout -k 5s "$duration" uv run --no-project python -B "$@" "${args[@]}" || rc=$?
  printf '%s\t%s\texit %d\n' "$(date -Iseconds)" "$name" "$rc" >> "$state"
  if (( rc > status )); then
    status=$rc
  fi
}

run_phase speed 4m "$validation" --phase speed
run_phase smoke 90s "$validation" --phase smoke
run_phase quality 3m "$validation" --phase quality --quality-cases \
  code-ttl code-lru episode-cache-first episode-cache-next retrieve-revision-1024
run_phase long-context 14m "$here/long_context.py" --trial "$4"
printf '%s\tbenchmark\texit %d\n' "$(date -Iseconds)" "$status" >> "$state"
exit "$status"
