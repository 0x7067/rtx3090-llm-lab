#!/usr/bin/env bash
set -euo pipefail

flashinfer_dir=$(/app/venv/bin/python -c 'import flashinfer, os; print(os.path.dirname(flashinfer.__file__))' | tail -n1)
files=("$flashinfer_dir/prefill.py" "$flashinfer_dir/decode.py")

# FlashInfer reuses these CPU planning workspaces across plan calls. Pageable
# memory makes the subsequent H2D copy synchronous with respect to the host,
# preventing the next MTP draft step from rewriting a pinned buffer whose copy
# is still in flight. Keep the patch narrow: sparse/MLA/POD buffers are outside
# the Qwen target/draft path under test.
matches=$(grep -h -c 'pin_memory=True' "${files[@]}" | awk '{sum += $1} END {print sum + 0}')
if [ "$matches" -ne 8 ]; then
  echo "expected 8 FlashInfer prefill/decode pinned planning buffers, found $matches" >&2
  exit 1
fi

sed -i 's/pin_memory=True/pin_memory=False/g' "${files[@]}"

if grep -q 'pin_memory=True' "${files[@]}"; then
  echo "FlashInfer planning-buffer patch is incomplete" >&2
  exit 1
fi

echo "patched 8 FlashInfer prefill/decode planning buffers to pageable memory"
