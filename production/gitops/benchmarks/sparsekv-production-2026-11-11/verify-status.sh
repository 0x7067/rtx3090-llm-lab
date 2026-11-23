#!/usr/bin/env bash
set -eu
here=$(cd -- "$(dirname -- "$0")" && pwd)
if [[ -f $here/checks.exit ]]; then
  printf 'Exit status: '
  cat "$here/checks.exit"
else
  echo 'Verification not finished'
fi
tail -n 10 "$here/checks.log"
