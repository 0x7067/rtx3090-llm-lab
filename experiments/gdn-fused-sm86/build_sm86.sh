#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-python3}

# The extension has cp.async and is intended for the RTX 3090 path.  Keep the
# architecture explicit so a successful build cannot accidentally target a
# different GPU family or emit a large multi-arch wheel.
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}

if [[ -z "${CUDA_HOME:-}" ]] && ! command -v nvcc >/dev/null 2>&1; then
  echo "gdn-fused-sm86 build requires nvcc or CUDA_HOME" >&2
  exit 3
fi

exec "${python_bin}" "${repo_dir}/setup.py" build_ext --inplace
