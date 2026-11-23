#!/bin/sh
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
systemctl --user show normal120-dflash-short-build-v1.service -p ActiveState -p SubState -p MainPID -p ExecMainStatus
test ! -f "$here/jobs/status.json" || cat "$here/jobs/status.json"
test ! -f "$here/jobs/build.log" || tail -n 8 "$here/jobs/build.log"
