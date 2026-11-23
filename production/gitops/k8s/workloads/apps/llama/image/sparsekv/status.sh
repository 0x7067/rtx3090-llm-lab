#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
systemctl --user show sparsekv-adoption-build-v1.service -p ActiveState -p SubState -p ExecMainStatus || true
[[ ! -f $here/qualification/build.exit ]] || cat "$here/qualification/build.exit"
tail -12 "$here/qualification/build.log"

if [[ -f $here/qualification/checks.exit ]]; then
  printf 'Repository checks exit: '
  cat "$here/qualification/checks.exit"
fi
