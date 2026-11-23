#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
systemctl --user show sparsekv-direct-checks -p ActiveState -p SubState -p ExecMainStatus
[[ ! -f $here/checks.exit ]] || cat "$here/checks.exit"
tail -10 "$here/checks.log"
