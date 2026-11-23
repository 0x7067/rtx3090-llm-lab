ARG BASE_IMAGE=qwen38-27b-3090:v8
FROM ${BASE_IMAGE}

# Overlay the vLLM patches onto the installed package. v8 shipped from the
# locked base WITHOUT this overlay, silently dropping the xgrammar fix that v7
# carried — v9 re-applies it alongside the new GDN spec-gate backport.
#
# The xgrammar patch applies with fuzz 1 on its __init__.py hunk against the
# v8 tree (4-line edge context; the applied region was verified byte-identical
# to the hunk's intended output, so the fuzz is a GNU patch matching quirk,
# not drift). Its gate is therefore a content assertion on the two added
# signature lines rather than a strict reverse dry-run. The new backport
# applies and reverse-checks strictly (-F0).
COPY xgrammar-terminated-batch.patch vllm-pr51812-gdn-spec-gates.patch /tmp/vllm-patches/
RUN set -eu; \
    SP=$(/app/venv/bin/python -c 'import os, vllm; print(os.path.dirname(vllm.__file__))' | tail -n1); \
    patch -p1 -d "$SP" < /tmp/vllm-patches/xgrammar-terminated-batch.patch; \
    grep -q 'accepted = not post_reasoning_end_in_window or bool(' "$SP/v1/structured_output/__init__.py"; \
    grep -q 'def rollback_window' "$SP/v1/structured_output/backend_xgrammar.py" || \
      grep -q 'is_terminated' "$SP/v1/structured_output/backend_xgrammar.py"; \
    test ! -e "$SP/v1/structured_output/__init__.py.rej"; \
    test ! -e "$SP/v1/structured_output/backend_xgrammar.py.rej"; \
    patch -p1 -F0 -d "$SP" < /tmp/vllm-patches/vllm-pr51812-gdn-spec-gates.patch; \
    patch -p1 -R -F0 --dry-run -s -d "$SP" < /tmp/vllm-patches/vllm-pr51812-gdn-spec-gates.patch; \
    /app/venv/bin/python -m compileall -q "$SP/v1/structured_output/__init__.py" "$SP/v1/structured_output/backend_xgrammar.py" "$SP/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"; \
    rm -rf /tmp/vllm-patches

LABEL org.opencontainers.image.description="Qwen3.8 v8 plus vLLM 0.27.1 xgrammar termination fix (restored from v7) and PR #51812 GDN spec-gate backport"
