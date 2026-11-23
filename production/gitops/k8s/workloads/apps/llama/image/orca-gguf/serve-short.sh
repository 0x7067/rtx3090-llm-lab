#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]{1,5}$ ]] || ((10#$1 < 1 || 10#$1 > 65535)); then
    echo 'Usage: serve-short.sh PORT' >&2
    exit 2
fi

export LD_LIBRARY_PATH=/opt/orca/llama-fast:/native-cuda

exec /opt/orca/llama-fast/llama-server \
    -m /opt/orca/weights/model.gguf --mmproj /opt/orca/weights/mmproj.gguf \
    --alias orcasaq2-cyber-fast --log-verbosity 4 --host 127.0.0.1 --port "$1" \
    --ctx-size 32768 --parallel 4 --gpu-layers 999 --fit off --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 --threads-batch 8 \
    --batch-size 2048 --ubatch-size 256 --jinja --no-context-shift --cont-batching --slots \
    --reasoning on --reasoning-budget 4096 --reasoning-effort xhigh --no-kv-unified \
    --spec-type draft-dflash2 --spec-draft-model /opt/orca/fast-draft.gguf --spec-draft-n-max 5 \
    --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 --spec-draft-ngl 999 \
    --metrics --no-cache-prompt --chat-template-file /opt/orca/chat_template.jinja
