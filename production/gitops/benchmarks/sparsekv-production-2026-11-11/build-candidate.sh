#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
scratch=$(cat "$here/.scratch")
image=127.0.0.1:5000/home-llm-strata@sha256:83077f60377d9e303d446a2f045381daab28a123e8e314c5bbe1ba53314884a2
site=/app/venv/lib/python3.12/site-packages

# The calling user owns the output archive; sudo is only for containerd.
# shellcheck disable=SC2024
sudo k3s ctr -n k8s.io run --rm --cpus 3 --memory-limit 8589934592 \
  --mount "type=bind,src=$scratch/hq,dst=/sparsebench,options=rbind:ro" \
  --env CUDA_VISIBLE_DEVICES= --env MAX_JOBS=3 \
  --env PIP_NO_INDEX=1 --env PIP_NO_CACHE_DIR=1 --env PY=/app/venv/bin/python \
  --env SITE="$site" \
  "$image" sparsekv-cpu-build /bin/bash -euc '
    cd /sparsebench
    bash pthfa/install.sh >&2
    bash sparsekv/install.sh /app/venv >&2
    bash sparsekv/install.sh /app/venv --check >&2
    cd "$SITE"
    tar -cf - pth_flash_attn pth_flash_attn-*.dist-info sparsekv \
      vllm/v1/core/kv_cache_utils.py vllm/v1/core/sched/scheduler.py \
      vllm/v1/worker/gpu/cudagraph_utils.py -C /app/venv share/sparsekv
  ' > "$scratch/overlay.tar.incomplete"
mv "$scratch/overlay.tar.incomplete" "$scratch/overlay.tar"
