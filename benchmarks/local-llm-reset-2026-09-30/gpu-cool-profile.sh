#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run with sudo" >&2
  exit 1
fi

case "${1:-}" in
  apply)
    systemctl is-active --quiet vram-thermal-guard.service
    systemctl is-active --quiet vram-fan-curve.service
    if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
      echo "GPU compute process is active" >&2
      exit 1
    fi
    nvidia-smi -i 0 --lock-gpu-clocks=210,1350
    systemctl stop gpu-mem-oc.service
    /opt/gpu-tune/venv/bin/python -c 'import pynvml as n; n.nvmlInit(); n.nvmlDeviceSetGpcClkVfOffset(n.nvmlDeviceGetHandleByIndex(0), 100)'
    offsets=$(/opt/gpu-tune/venv/bin/python -c 'import pynvml as n; n.nvmlInit(); d=n.nvmlDeviceGetHandleByIndex(0); print(n.nvmlDeviceGetGpcClkVfOffset(d), n.nvmlDeviceGetMemClkVfOffset(d))')
    if [[ "$offsets" != "100 0" ]]; then
      echo "Graphics and memory VF offsets are $offsets, expected 100 0" >&2
      exit 1
    fi
    rm -f /run/gpu-thermal-trip
    ;;
  restore)
    if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
      echo "GPU compute process is active" >&2
      exit 1
    fi
    /opt/gpu-tune/venv/bin/python -c 'import pynvml as n; n.nvmlInit(); n.nvmlDeviceSetGpcClkVfOffset(n.nvmlDeviceGetHandleByIndex(0), 0)'
    nvidia-smi -i 0 --reset-gpu-clocks
    systemctl start gpu-mem-oc.service
    ;;
  *)
    echo "Usage: $0 apply|restore" >&2
    exit 2
    ;;
esac

nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.limit,power.draw --format=csv,noheader
/opt/gpu-tune/venv/bin/python -c 'import pynvml as n; n.nvmlInit(); d=n.nvmlDeviceGetHandleByIndex(0); print("graphics VF offset:", n.nvmlDeviceGetGpcClkVfOffset(d), "memory VF offset:", n.nvmlDeviceGetMemClkVfOffset(d))'
systemctl is-active vram-thermal-guard.service vram-fan-curve.service
systemctl is-active gpu-mem-oc.service || true
