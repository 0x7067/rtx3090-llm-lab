#!/usr/bin/env bash
# Run the OBLITERATUS/Qwen3.8-27B-OBLITERATED Q4_K_M GGUF standalone on
# :8095 as the second qwen option. This uses the llama.cpp lane because the
# repo ships no W4A16 checkpoint for the vLLM image. Assumes the GPU is already
# free (deployment scaled to 0; see bench_window.sh).
# Patched MTP n-max 3 reached 64.1 tok/s vs 42.4 off (+51%). n-max 5 fell to
# 59.4 tok/s as acceptance dropped. Mainline drops MTP tensors and cannot
# speculate. Patched vision uses ~22.1 GB of 24 GB at NP=4, CTX=65536, q8_0 KV.
set -euo pipefail
cd "$(dirname "$0")"

CTX="${CTX:-65536}"
NP="${NP:-4}"
MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf}"
VISION="${VISION:-1}"
IMAGE="${IMAGE:-llama:cuda-swap-v13-kvarn-rc2}"
DEFAULT_SPEC_ARGS="--spec-type draft-mtp --spec-draft-n-max 3"
SPEC_ARGS="${SPEC_ARGS-$DEFAULT_SPEC_ARGS}"
KV_TYPE="${KV_TYPE:-q8_0}"
METRICS="${METRICS:-1}"

DOCKER_ARGS=()
ARGS=()
# Mainline dispatches through its default CMD; the patched image exposes llama-server directly.
if [[ "$IMAGE" == *ggml-org* ]]; then
  if [[ "$SPEC_ARGS" == "$DEFAULT_SPEC_ARGS" ]]; then
    echo "warning: $IMAGE cannot use MTP; clearing default SPEC_ARGS" >&2
    SPEC_ARGS=""
  fi
  ARGS+=(--server)
else
  DOCKER_ARGS+=(--entrypoint /usr/local/bin/llama-server)
fi
ARGS+=(
  -m "/models/$MODEL_FILE"
  -ngl 99 -c "$CTX" -np "$NP" --jinja
  --host 0.0.0.0 --port 8080
)
[ "$VISION" = "0" ] || ARGS+=(--mmproj /models/mmproj-model-bf16.gguf)
[ -z "$KV_TYPE" ] || ARGS+=(--cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE")
[ "$METRICS" = "0" ] || ARGS+=(--metrics)
# Mainline has no --spec-type; SPEC_ARGS is only valid with the patched image.
[ -z "$SPEC_ARGS" ] || ARGS+=($SPEC_ARGS)

docker rm -f oblitbench >/dev/null 2>&1 || true
echo "IMAGE=$IMAGE CTX=$CTX NP=$NP SPEC_ARGS=$SPEC_ARGS KV_TYPE=$KV_TYPE"
docker run -d --name oblitbench --gpus all \
  -p 127.0.0.1:8095:8080 \
  -v "$PWD/models:/models:ro" \
  "${DOCKER_ARGS[@]}" \
  "$IMAGE" \
  "${ARGS[@]}"

# llama.cpp can take several minutes to load the model and vision projector.
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 2 http://127.0.0.1:8095/health || true)
  [ "$code" = "200" ] && { echo "OBLIT READY after $((i*5))s"; exit 0; }
  state=$(docker inspect -f '{{.State.Running}}' oblitbench 2>/dev/null || echo false)
  [ "$state" = "false" ] && { echo "OBLIT DIED"; docker logs --tail 30 oblitbench; exit 1; }
  sleep 5
done
echo "OBLIT TIMEOUT"; docker logs --tail 30 oblitbench; exit 1
