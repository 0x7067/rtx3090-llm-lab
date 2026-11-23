FROM 127.0.0.1:5000/home-llm-strata:normal120-dflash-short-v1@sha256:83077f60377d9e303d446a2f045381daab28a123e8e314c5bbe1ba53314884a2
COPY --chmod=644 runtime.py kernels.py /app/venv/lib/python3.12/site-packages/vllm_compact_gdn/
COPY --chmod=644 apply_patch.py source-hashes.json /opt/compact-gdn/
RUN chmod 755 /app/venv/lib/python3.12/site-packages/vllm_compact_gdn \
    && touch /app/venv/lib/python3.12/site-packages/vllm_compact_gdn/__init__.py \
    && /app/venv/bin/python -I -S /opt/compact-gdn/apply_patch.py /app/venv/lib/python3.12/site-packages/vllm
