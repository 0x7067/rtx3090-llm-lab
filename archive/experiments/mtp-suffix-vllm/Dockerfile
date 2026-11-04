ARG BASE_IMAGE=qwen38-27b-3090:v9
FROM ${BASE_IMAGE}

# Isolated derivative only. The runtime remains unchanged unless the caller
# opts into VLLM_MTP_SUFFIX_LOOKUP=1 and configures a seven-token max block.
USER root
COPY mtp-suffix-vllm.patch /tmp/mtp-suffix-vllm.patch
RUN set -eu; \
    patch -p1 -F0 -d /app/venv/lib/python3.12/site-packages/vllm < /tmp/mtp-suffix-vllm.patch; \
    /app/venv/bin/python -m compileall -q \
      /app/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py \
      /app/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/cudagraph_utils.py; \
    patch -p1 -R -F0 --dry-run -s -d /app/venv/lib/python3.12/site-packages/vllm < /tmp/mtp-suffix-vllm.patch; \
    rm -f /tmp/mtp-suffix-vllm.patch

LABEL org.opencontainers.image.description="v9 plus opt-in MTP suffix lookup experiment"
