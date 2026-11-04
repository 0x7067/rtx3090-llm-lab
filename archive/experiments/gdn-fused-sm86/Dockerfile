FROM qwen38-27b-3090:v9

USER root
RUN mkdir -p /app/experimental
COPY experiments/gdn-fused-sm86/gdn_fused_sm86_ext.so /app/experimental/gdn_fused_sm86_ext.so
COPY experiments/gdn-fused-sm86/patch_vllm.py /tmp/patch_vllm.py
RUN /app/venv/bin/python /tmp/patch_vllm.py /app/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py \
    && /app/venv/bin/python -m py_compile /app/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py \
    && rm /tmp/patch_vllm.py

USER 1000:1000
