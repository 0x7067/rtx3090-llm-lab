ARG BASE_IMAGE=qwen38-27b-3090:v9
FROM ${BASE_IMAGE}

# Isolated derivative only. The runtime remains unchanged unless the caller
# opts into VLLM_MTP_SUFFIX_LOOKUP=1 with synchronous scheduling.
USER root
COPY mtp-suffix-vllm-v5.patch /tmp/mtp-suffix-vllm-v5.patch
RUN set -eu; \
    patch -p1 -F0 -d /app/venv/lib/python3.12/site-packages/vllm < /tmp/mtp-suffix-vllm-v5.patch; \
    /app/venv/bin/python -m compileall -q \
      /app/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/llm_base_proposer.py; \
    patch -p1 -R -F0 --dry-run -s -d /app/venv/lib/python3.12/site-packages/vllm < /tmp/mtp-suffix-vllm-v5.patch; \
    rm -f /tmp/mtp-suffix-vllm-v5.patch

LABEL org.opencontainers.image.description="v9 plus opt-in legacy-runner MTP suffix lookup experiment"
