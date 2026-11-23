#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
scratch=$(cat "$here/.scratch")
if [[ -f $scratch/build.exit ]]; then
  printf 'exit status: '
  cat "$scratch/build.exit"
elif kill -0 "$(cat "$scratch/build.pid")" 2>/dev/null; then
  echo 'running'
else
  echo 'build stopped without recording an exit status'
fi
tail -n 22 "$scratch/build.log"
