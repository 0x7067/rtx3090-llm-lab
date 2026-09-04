#!/usr/bin/env bash
# imatrix on GPU from BF16, then CPU quants. Run only while the GPU is free.
set -euo pipefail
IMG="${IMG:-llama:cuda-swap-v16}"
M=/data/buttercup_6tb/k3s/llama-models/k2-horizon
BF16=/m/IFM/K2-Horizon-7B-GGUF/K2-Horizon-7B-BF16.gguf
run() { docker run --rm --gpus all -u 1000:1000 -v "$M":/m --entrypoint "$1" "$IMG" "${@:2}"; }
if [ ! -s "$M/local/K2-Horizon-7B.imatrix" ]; then
  run /usr/local/bin/llama-imatrix -m "$BF16" -f /m/local/calib-code-chat.txt -o /m/local/K2-Horizon-7B.imatrix -ngl 99 -c 2048 -b 2048 -ub 512 --chunks 280 -fa on 2>&1 | grep -E "compute_imatrix|save_imatrix|error|tokens" | tail -8
fi
for q in Q6_K Q5_K_M Q4_K_M; do
  [ -s "$M/local/K2-Horizon-7B-$q.gguf" ] && continue
  run /usr/local/bin/llama-quantize --imatrix /m/local/K2-Horizon-7B.imatrix "$BF16" "/m/local/K2-Horizon-7B-$q.gguf" "$q" 12 2>&1 | grep -E "quant size|error" 
done
ls -la "$M/local/"
