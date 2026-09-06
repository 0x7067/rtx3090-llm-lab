#!/usr/bin/env bash
# Export the trained EAGLE-3 draft to HF layout, then convert it to GGUF for
# llama.cpp. CPU only - safe to run while the local API is serving.
#
# The GGUF converter needs a llama.cpp source tree; the runtime image ships
# only binaries. CONVERT_SRC is a checkout of the image's pinned ref with
# patches-v15 applied, which is what supplies conversion/k2_horizon.py for the
# target's vocab handler. Upstream already routes LlamaForCausalLMEagle3
# through the Llama converter.
set -euo pipefail
W=/data/buttercup_6tb/specforge-work
E=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FEATURES="${FEATURES:-$W/cache/hidden_states/k2-7b-eagle3-regen}"
CKPT="${CKPT:-$W/outputs/k2-horizon-7b-eagle3-regen/k2-horizon-7b-eagle3-regen-latest}"
EXPORT_DIR="${EXPORT_DIR:-$W/exports/k2-7b-eagle3-regen-hf}"
CONVERT_SRC="${CONVERT_SRC:-$W/llama.cpp-convert}"
OUT="${OUT:-/data/buttercup_6tb/k3s/llama-models/k2-horizon/local/K2-Horizon-7B-Eagle3-Q8_0.gguf}"
TARGET="$W/models/IFM/K2-Horizon-7B"

# Its own lock, not the training lock: this script is deliberately runnable
# while training holds the GPU, which is how the path was proven against the
# step-8000 checkpoint. What needs guarding is EXPORT_DIR, which the rm -rf
# below wipes -- a second concurrent run would delete the first one's export
# mid-conversion.
exec 8>"$W/.k2-export.lock"
flock -n 8 || { echo 'Another export/convert run owns the lock' >&2; exit 1; }

[[ -d "$CKPT" ]] || { echo "No checkpoint at $CKPT" >&2; exit 1; }
[[ -f "$CONVERT_SRC/convert_hf_to_gguf.py" ]] || { echo "No converter at $CONVERT_SRC" >&2; exit 1; }
export HF_HOME="$W/hf-home" HF_HUB_OFFLINE=1
# shellcheck source=/dev/null
source "$W/venv/bin/activate"

rm -rf "$EXPORT_DIR"
specforge export --to hf \
  --checkpoint "$CKPT" \
  --draft-config "$E/configs/k2-horizon-7b-eagle3.json" \
  --output-dir "$EXPORT_DIR" \
  --vocab-mapping "$FEATURES/vocab_mapping/vocab_mapping.pt" \
  --embedding-source "$TARGET"

mkdir -p "$(dirname "$OUT")"
PYTHONPATH="$CONVERT_SRC/gguf-py" python "$CONVERT_SRC/convert_hf_to_gguf.py" "$EXPORT_DIR" \
  --target-model-dir "$TARGET" \
  --outtype q8_0 \
  --outfile "$OUT"
ls -l "$OUT"
echo EXPORT-CONVERT-DONE
